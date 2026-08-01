from manim import *
import numpy as np

# Base class provided by the framework
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
        # Data from storyboard
        title_text = "The Grand Hook: A Meeting of Giants"
        lecture_lines = [
            "Euler's identity is the most beautiful equation in mathematics.",
            "It connects five fundamental constants in one elegant line.",
            "Zero and one represent the basics of arithmetic.",
            "Pi and e emerge from geometry and calculus.",
            "The imaginary unit i joins them in perfect harmony."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define constants as Text/MarkupText
        const_e = Text("e", font_size=60, slant=ITALIC)
        const_i = Text("i", font_size=60, slant=ITALIC)
        const_pi = Text("π", font_size=60)
        const_1 = Text("1", font_size=60)
        const_0 = Text("0", font_size=60)
        
        # Initial placement using grid
        # Fix for Issue 24: Move const_0 from C6 to C5
        self.place_at_grid(const_e, "B2")
        self.place_at_grid(const_i, "B5")
        self.place_at_grid(const_pi, "E2")
        self.place_at_grid(const_1, "E5")
        self.place_at_grid(const_0, "C5", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(VGroup(const_e, const_i, const_pi, const_1, const_0)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFFE0")
        )
        
        ei_group_target = VGroup(const_e.copy(), const_i.copy()).arrange(RIGHT, buff=0.2)
        # Fix for Issue 25: Change area for ei_group_target to B3-D4
        self.place_in_area(ei_group_target, "B3", "D4", scale_factor=1.0)
        center_pos = ei_group_target.get_center()
        
        glow = Dot(center_pos, radius=1.2, color="#FFFFE0").set_opacity(0.2)
        
        self.play(
            const_e.animate.move_to(ei_group_target[0]).set_color("#FFFFE0"),
            const_i.animate.move_to(ei_group_target[1]).set_color("#FFFFE0"),
            FadeIn(glow)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFA500")
        )
        
        path = Circle(radius=1.3, color="#FFA500").move_to(center_pos)
        self.play(Create(path), const_pi.animate.set_color("#FFA500"))
        
        self.play(const_pi.animate.move_to(path.point_from_proportion(0)))
        self.play(MoveAlongPath(const_pi, path), run_time=2.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#90EE90")
        )
        
        self.play(
            const_1.animate.move_to(self.grid["D4"]).set_color("#90EE90"),
            const_0.animate.move_to(self.grid["D5"]).set_color("#ADD8E6")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        
        # Use MarkupText for the final equation to handle superscripts without LaTeX
        equation = MarkupText("<i>e</i><sup><i>i</i>π</sup> + 1 = 0", font_size=80, color=WHITE)
        # Fix for Issue 26: Change area for equation to C2-D5
        self.place_in_area(equation, "C2", "D5", scale_factor=1.0)
        
        self.play(
            FadeOut(glow),
            FadeOut(path),
            ReplacementTransform(VGroup(const_e, const_i, const_pi, const_1, const_0), equation)
        )
        self.wait(3)
