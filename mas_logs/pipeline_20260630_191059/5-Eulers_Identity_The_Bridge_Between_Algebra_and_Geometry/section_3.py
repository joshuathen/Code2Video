from manim import *
import numpy as np

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
        # Title and lecture lines from storyboard
        title_text = "The Engine of Growth: Understanding 'e'"
        lecture_lines = [
            "The number e represents continuous growth in one direction.",
            "The imaginary unit i forces growth to become perpendicular.",
            "This constant sideways push creates a perfect circular path."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors from storyboard
        GOLD = "#FFD700"
        BLUE_I = "#1E90FF"
        MAGENTA = "#FF00FF"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show 'e' in gold (#FFD700) with a horizontal growth vector.
        self.play(self.lecture[0].animate.set_color(GOLD))
        
        # 'e' label at D1
        e_label = Text("e", font_size=48, color=GOLD)
        self.place_at_grid(e_label, "D1", scale_factor=1.0)
        
        # Horizontal growth vector at D2 (Issue 27 Fix)
        yellow_arrow = Arrow(LEFT, RIGHT, color=GOLD, buff=0)
        self.place_at_grid(yellow_arrow, "D2", scale_factor=1.0)
        
        self.play(Write(e_label), GrowArrow(yellow_arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce 'i' as a perpendicular blue force vector (#1E90FF) at the vector tip.
        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            self.lecture[1].animate.set_color(BLUE_I)
        )
        
        # Perpendicular vector 'i' at C3 (Issue 28 Fix)
        blue_arrow = Arrow(DOWN, UP, color=BLUE_I, buff=0)
        self.place_at_grid(blue_arrow, "C3", scale_factor=0.8)
        
        i_label = Text("i", font_size=48, color=BLUE_I, slant=ITALIC)
        self.place_at_grid(i_label, "B3", scale_factor=0.8)
        
        self.play(GrowArrow(blue_arrow), Write(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The growth path bends 90 degrees into a magenta circular arc (#FF00FF).
        # Animate the vector tip tracing a full circle in magenta (#FF00FF).
        # Highlight the final circle in white (#FFFFFF) and label it 'Circular Growth'.
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            self.lecture[2].animate.set_color(MAGENTA)
        )
        
        # Circular path in area B4-E6 (Issue 29 Fix)
        growth_path = Circle(radius=1.5, color=MAGENTA)
        self.place_in_area(growth_path, "B4", "E6", scale_factor=0.9)
        center = growth_path.get_center()
        radius = growth_path.radius
        
        # Rotating vector for tracing
        rotating_vector = Arrow(center, center + RIGHT * radius, color=GOLD, buff=0)
        
        # Fade out static elements and prepare tracing
        self.play(FadeOut(yellow_arrow), FadeOut(blue_arrow), FadeOut(e_label), FadeOut(i_label))
        self.play(FadeIn(rotating_vector))
        
        # Animate the vector tracing a full circle
        self.play(
            Rotate(rotating_vector, angle=TAU, about_point=center),
            Create(growth_path),
            run_time=4,
            rate_func=linear
        )
        
        # Final highlight and label
        self.play(growth_path.animate.set_color(WHITE_COLOR))
        
        label_circular = Text("Circular Growth", font_size=24, color=WHITE_COLOR)
        self.place_at_grid(label_circular, "F5", scale_factor=1.0)
        self.play(Write(label_circular))
        
        self.wait(2)
