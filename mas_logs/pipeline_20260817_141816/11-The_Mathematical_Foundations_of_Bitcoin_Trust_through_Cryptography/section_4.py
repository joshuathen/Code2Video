from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Public-Key Cryptography: Digital Signatures", [
            "Users own a private key for signatures.",
            "Public keys verify ownership without revealing secrets.",
            "Cryptography ensures secure transaction validation."
        ])

        # Assets
        private_key = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lock.svg", color=BLUE)
        private_label = Text("Private Key", font_size=18).next_to(private_key, DOWN, buff=0.1)
        public_key = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg", color=YELLOW)
        public_label = Text("Public Key", font_size=18).next_to(public_key, DOWN, buff=0.1)
        
        message = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/envelope.svg", color=WHITE)
        signature = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg", color="#9B59B6")
        sig_label = Text("Signature", font_size=18, color="#9B59B6").next_to(signature, DOWN, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(private_key, 'C2', scale_factor=0.8)
        self.place_at_grid(private_label, 'C2')
        private_label.shift(DOWN * 0.5)
        self.play(FadeIn(private_key), Write(private_label))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[0].set_color(GRAY)
        self.place_at_grid(public_key, 'C5', scale_factor=0.8)
        self.place_at_grid(public_label, 'C5')
        public_label.shift(DOWN * 0.5)
        self.play(FadeIn(public_key), Write(public_label))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[1].set_color(GRAY)
        
        self.place_at_grid(message, 'E2', scale_factor=0.7)
        self.place_at_grid(signature, 'E5', scale_factor=0.7)
        self.place_at_grid(sig_label, 'E5')
        sig_label.shift(DOWN * 0.5)
        
        self.play(Write(message))
        self.play(message.animate.set_color("#9B59B6"), FadeIn(signature), Write(sig_label))
        self.play(signature.animate.set_color("#2ECC71"), sig_label.animate.set_color("#2ECC71"))
        
        checkmark = Tex(r"$\checkmark$", color=GREEN, font_size=40)
        self.place_at_grid(checkmark, 'F5', scale_factor=0.6)
        self.play(Write(checkmark))
        self.wait(2)
