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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "States define systems by specific values.",
            "Classical systems are binary, like switches.",
            "Quantum systems allow multiple potential states.",
            "Hilbert space maps these quantum possibilities.",
            "A spinning coin represents potential outcomes."
        ]
        self.setup_layout("The Classical vs. Quantum Prelude", lecture_lines)
        
        # Elements
        classical_text = Text("Classical State", color=WHITE)
        quantum_text = Text("Quantum State", color=YELLOW)
        
        # Assets
        switch_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        # Pulsating effect for the coin
        coin_icon.add_updater(lambda m, dt: m.set_opacity(0.5 + 0.5 * np.sin(self.time * 2)))

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(classical_text, 'C2', scale_factor=0.6)
        self.place_at_grid(quantum_text, 'C5', scale_factor=0.6)
        self.place_at_grid(switch_icon, 'B2', scale_factor=0.5)
        self.play(FadeIn(classical_text), FadeIn(quantum_text), FadeIn(switch_icon))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(switch_icon.animate.shift(RIGHT * 0.2), run_time=0.5)
        self.play(switch_icon.animate.shift(LEFT * 0.2), run_time=0.5)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(coin_icon, 'D5', scale_factor=0.3)
        self.play(FadeIn(coin_icon), self.lecture[2].animate.set_color(BLUE))

        # === Animation for Lecture Line 4 ===
        # Representing Hilbert Space as a small box/area
        hilbert_space = Square(side_length=1.5, color=PURPLE)
        self.place_in_area(hilbert_space, 'E4', 'E5', scale_factor=0.5)
        self.play(Create(hilbert_space), self.lecture[3].animate.set_color(BLUE))

        # === Animation for Lecture Line 5 ===
        # Final color change highlight
        self.play(
            classical_text.animate.set_color(BLUE),
            quantum_text.animate.set_color(BLUE),
            self.lecture[4].animate.set_color(BLUE)
        )
        self.wait(2)
