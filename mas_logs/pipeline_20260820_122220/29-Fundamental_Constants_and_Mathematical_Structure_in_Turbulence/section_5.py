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
        self.setup_layout("Synthesis and Summary", [
            "Statistical laws simplify complex flow modeling.", 
            "We predict turbulence without tracking particles.", 
            "Order exists within chaotic statistical systems."
        ])
        
        # Elements
        summary_box = RoundedRectangle(corner_radius=0.2, height=3, width=4, color=BLUE)
        summary_text = VGroup(
            Text("Statistical Synthesis", font_size=24, color=BLUE),
            Text("Modeling simplified", font_size=20, color=WHITE),
            Text("via cascade laws.", font_size=20, color=WHITE)
        ).arrange(DOWN)
        summary_obj = VGroup(summary_box, summary_text)
        
        ck_text = Tex(r"$C_k$", font_size=40, color=YELLOW)
        ck_label = Text("Kolmogorov Constant", font_size=18, color=YELLOW)
        ck_group = VGroup(ck_text, ck_label).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(summary_obj, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(summary_obj))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(ck_group, 'D4', scale_factor=0.7)
        self.play(Write(ck_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(FadeOut(summary_obj), FadeOut(ck_group))
        
        final_text = Text("Turbulence: Order in Chaos", font_size=32, color=GREEN)
        self.place_at_grid(final_text, 'E4', scale_factor=0.8)
        self.play(GrowFromCenter(final_text))
        
        self.wait(2)
        self.play(FadeOut(self.title), FadeOut(self.lecture), FadeOut(final_text))
