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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Integration and differentiation are inverses.",
            "Summing pieces reverses breaking down.",
            "The function machine confirms this.",
            "Differentiation undoes the integration process.",
            "They form a perfect connection."
        ]
        self.setup_layout("The Fundamental Theorem: Connecting the Dots", lecture_lines)

        # Assets
        int_symbol = MathTex(r"\int f(x) \, dx", font_size=48, color=YELLOW)
        der_symbol = MathTex(r"\frac{d}{dx} F(x)", font_size=48, color=BLUE)
        
        # Correctly using the provided asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg
        machine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg", color=WHITE)
        machine_label = Text("Function Machine", font_size=24)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Applying fix: self.place_at_grid(int_symbol, 'B3', scale_factor=1.0); self.place_at_grid(der_symbol, 'B4', scale_factor=1.0)
        self.place_at_grid(int_symbol, "B3", scale_factor=1.0)
        self.place_at_grid(der_symbol, "B4", scale_factor=1.0)
        self.play(Write(int_symbol), Write(der_symbol))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        area_rect = Rectangle(height=1, width=1, fill_opacity=0.5, color=GREEN).set_fill(GREEN)
        self.place_at_grid(area_rect, "D2")
        self.play(FadeIn(area_rect), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        # Applying fix: self.place_in_area(machine, 'C4', 'E5', scale_factor=0.9)
        self.place_in_area(machine, "C4", "E5", scale_factor=0.9)
        # Applying fix: self.place_at_grid(machine_label, 'E4', scale_factor=0.8)
        self.place_at_grid(machine_label, "E4", scale_factor=0.8)
        self.play(Create(machine), Write(machine_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        self.play(Indicate(int_symbol), Indicate(der_symbol), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        self.play(FadeOut(int_symbol), FadeOut(der_symbol),
                  FadeOut(area_rect), FadeOut(machine), FadeOut(machine_label))
        self.wait(1)
