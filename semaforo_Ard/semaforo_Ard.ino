// --- PINOS DOS SEMÁFOROS ---
// Pinos 1: Controlam Norte e Sul juntos (S1/S3)
const int Vm1 = 7;
const int Am1 = 6;
const int Vd1 = 5;

// Pinos 2: Controlam Leste e Oeste juntos (S2/S4)
const int Vm2 = 4; 
const int Am2 = 3; 
const int Vd2 = 2;

// Variáveis de controle de tempo (Substitutos do delay)
unsigned long tempoAnterior = 0;
int faseNormal = 0;

// String para armazenar o comando da IA
String comandoIA = "XX";

void setup() {
  Serial.begin(9600); // Abre a comunicação serial
  pinMode(Vm1, OUTPUT); pinMode(Am1, OUTPUT); pinMode(Vd1, OUTPUT);
  pinMode(Vm2, OUTPUT); pinMode(Am2, OUTPUT); pinMode(Vd2, OUTPUT);
  Serial.println("Sistema Inteligente Pronto! Envie os comandos (Ex: PN, VL, XX):");
}

void loop() {
  // 1. LEITURA DA IA VIA SERIAL
  if (Serial.available() > 0) {
    String leitura = Serial.readString();
    leitura.trim();
    leitura.toUpperCase();
    if (leitura == "XX" || leitura == "X" || leitura.length() == 2) {
      comandoIA = leitura;
      Serial.print("Comando Ativo: ");
      Serial.println(comandoIA);
    }
  }

  // 2. HIERARQUIA DE DECISÃO INTELIGENTE
  if (comandoIA == "XX" || comandoIA == "X") {
    rotinaNormal();
  }
  else if (comandoIA.startsWith("P")) {
    char localPedestre = comandoIA[1];
    if (localPedestre == 'N' || localPedestre == 'S') {
      fecharNorteSul_AbrirLesteOeste();
    }
    else if (localPedestre == 'L' || localPedestre == 'O') {
      fecharLesteOeste_AbrirNorteSul();
    }
  } 
  else if (comandoIA.startsWith("V")) {
    char direcaoAmbulancia = comandoIA[1];
    modoEmergencia(direcaoAmbulancia);
  }
}

// --- FUNÇÕES DE CONTROLE DE FLUXO ---

// Pedestre no Norte ou Sul: Fecha N/S e Abre L/O
void fecharNorteSul_AbrirLesteOeste() {
  digitalWrite(Vd1, LOW); digitalWrite(Am1, LOW); digitalWrite(Vm1, HIGH);
  digitalWrite(Vm2, LOW); digitalWrite(Am2, LOW); digitalWrite(Vd2, HIGH);
}

// Pedestre no Leste ou Oeste: Fecha L/O e Abre N/S
void fecharLesteOeste_AbrirNorteSul() {
  digitalWrite(Vd2, LOW); digitalWrite(Am2, LOW); digitalWrite(Vm2, HIGH);
  digitalWrite(Vm1, LOW); digitalWrite(Am1, LOW); digitalWrite(Vd1, HIGH);
}

// Modo de emergência (Veículos de Emergência / Ambulância)
void modoEmergencia(char local) {
  // Coloca ambas as vias em Vermelho primeiro

  // Abre a via correspondente à emergência
  if (local == 'N' || local == 'S') {
    digitalWrite(Vd2, LOW); digitalWrite(Am2, LOW); digitalWrite(Vm2, HIGH);
    digitalWrite(Vm1, LOW); digitalWrite(Am1, LOW);digitalWrite(Vd1, HIGH);
  }
  else if (local == 'L' || local == 'O') {
    digitalWrite(Vd1, LOW); digitalWrite(Am1, LOW); digitalWrite(Vm1, HIGH);
    digitalWrite(Vm2, LOW); digitalWrite(Am2, LOW);digitalWrite(Vd2, HIGH);
  }
}

// Ciclo normal estruturado com millis
void rotinaNormal() {
  unsigned long tempoAtual = millis();
  switch(faseNormal) {
    case 0: // Norte-Sul VERDE | Leste-Oeste VERMELHO
      digitalWrite(Vd1, HIGH); digitalWrite(Am1, LOW); digitalWrite(Vm1, LOW);
      digitalWrite(Vd2, LOW); digitalWrite(Am2, LOW); digitalWrite(Vm2, HIGH);
      if (tempoAtual - tempoAnterior >= 6000) { faseNormal = 1; tempoAnterior = tempoAtual; }
      break;
    case 1: // Norte-Sul AMARELO | Leste-Oeste VERMELHO
      digitalWrite(Vd1, LOW); digitalWrite(Am1, HIGH);
      if (tempoAtual - tempoAnterior >= 2000) { faseNormal = 2; tempoAnterior = tempoAtual; }
      break;
    case 2: // Norte-Sul VERMELHO | Leste-Oeste VERDE
      digitalWrite(Vd1, LOW); digitalWrite(Am1, LOW); digitalWrite(Vm1, HIGH);
      digitalWrite(Vd2, HIGH); digitalWrite(Am2, LOW); digitalWrite(Vm2, LOW);
      if (tempoAtual - tempoAnterior >= 6000) { faseNormal = 3; tempoAnterior = tempoAtual; }
      break;
    case 3: // Norte-Sul VERMELHO | Leste-Oeste AMARELO
      digitalWrite(Vd2, LOW); digitalWrite(Am2, HIGH);
      if (tempoAtual - tempoAnterior >= 2000) { faseNormal = 0; tempoAnterior = tempoAtual; }
      break;
  }
}