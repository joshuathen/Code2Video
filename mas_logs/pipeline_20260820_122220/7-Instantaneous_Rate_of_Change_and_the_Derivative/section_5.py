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
        lecture_lines = ["Derivative is the slope function.", "It captures instantaneous change.", "Secant converges to tangent."]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # Consistent color theme
        HIGHLIGHT_COLOR = "#00FFFF"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1), run_time=0.5)
        # Using placeholder for SVG as per instruction - Note: none.svg is empty/not functional as a visual
        formula_derivative = Text("Derivative = Instantaneous Slope", font_size=28, color=WHITE)
        self.place_at_grid(formula_derivative, 'C4', scale_factor=0.8)
        self.play(Write(formula_derivative))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_opacity(1), run_time=0.5)
        # Highlight logic
        derivative_label = Text("Derivative", font_size=24, color=HIGHLIGHT_COLOR).next_to(formula_derivative, UP)
        slope_label = Text("Slope", font_size=24, color=HIGHLIGHT_COLOR).next_to(formula_derivative, DOWN)
        self.play(Write(derivative_label), Write(slope_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_opacity(1), run_time=0.5)
        transition_labels = VGroup(
            Text("Secant", font_size=20, color=WHITE),
            Text("->", font_size=20, color=WHITE),
            Text("Tangent", font_size=20, color=WHITE)
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(transition_labels, 'D4', 'D5', scale_factor=0.75)
        self.play(FadeIn(transition_labels))
        
        self.wait(2)
        self.play(FadeOut(self.title), FadeOut(self.lecture), FadeOut(formula_derivative), FadeOut(derivative_label), FadeOut(slope_label), FadeOut(transition_labels))
