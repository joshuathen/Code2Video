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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Intuition", [
            "Conservation laws compute transcendental constants.",
            "Physical systems act as analog computers.",
            "Pi emerges from simple collision rules."
        ])
        
        billiard_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg"
        
        # === Animation for Lecture Line 1 ===
        # Fade in a visual summary of the system
        summary_group = VGroup(
            SVGMobject(billiard_path, color=WHITE),
            Square(color=BLUE),
            Square(color=RED).shift(RIGHT * 2)
        )
        self.place_in_area(summary_group, 'B2', 'C5', scale_factor=0.5)
        self.play(FadeIn(summary_group), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Flash the connection between physics and numbers
        connect_text = Text("Physics -> Number", font_size=32, color=GREEN)
        self.place_at_grid(connect_text, 'D3', scale_factor=0.6)
        self.play(Write(connect_text), self.lecture[1].animate.set_color("#00FF00"))
        self.play(Indicate(connect_text))

        # === Animation for Lecture Line 3 ===
        # Pi emerges and fade to black
        pi_text = MathTex(r"\pi \approx 3.14159...", color=GOLD)
        self.place_at_grid(pi_text, 'D4', scale_factor=0.7)
        self.play(FadeIn(pi_text), self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(1)
        
        billiard_final = SVGMobject(billiard_path, color=BLACK)
        self.place_in_area(billiard_final, 'B2', 'C5', scale_factor=0.5)
        self.play(FadeOut(summary_group), FadeOut(connect_text), FadeOut(pi_text), FadeIn(billiard_final))
        self.play(FadeOut(billiard_final))
