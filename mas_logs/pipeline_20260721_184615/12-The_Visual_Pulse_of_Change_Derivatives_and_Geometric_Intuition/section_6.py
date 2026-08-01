from manim import *
import numpy as np

# === Base Class ===
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

# === Section 6 Scene ===
class Section6Scene(TeachingScene):
    def construct(self):
        # Setup
        title_text = "Summary & Real-World Bridge"
        lecture_lines = [
            "Derivatives represent slopes and instantaneous rates of change.",
            "They bridge the gap between algebra and geometric intuition.",
            "This allows us to track exact velocity in real-time."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # 1. Geometric visual: Red Tangent Line (#FF0000)
        axes = Axes(
            x_range=[0, 2, 1], 
            y_range=[0, 2, 1], 
            x_length=2.0, 
            y_length=2.0,
            axis_config={"color": "#FFFFFF", "include_tip": False}
        )
        curve = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 1.8], color="#FFFFFF")
        tangent_point = axes.c2p(1.2, 0.5 * 1.2**2)
        dot = Dot(tangent_point, color="#FF0000")
        # Approximation of tangent line at x=1.2 (slope = 1.2)
        tangent_line = Line(
            axes.c2p(0.6, 0.5 * 1.2**2 - 0.6 * 1.2), 
            axes.c2p(1.8, 0.5 * 1.2**2 + 0.6 * 1.2), 
            color="#FF0000"
        )
        geo_visual = VGroup(axes, curve, dot, tangent_line)
        # Resolved Issue 33: Moving row A up to avoid bottom-heavy look
        self.place_in_area(geo_visual, "A1", "C3", scale_factor=0.8)

        # 2. Algebraic visual: Cyan Power Rule (#00FFFF)
        algebra_visual = MathTex(r"\frac{d}{dx} x^n = nx^{n-1}", color="#00FFFF")
        # Resolved Issue 33: Moving row A up to avoid bottom-heavy look
        self.place_in_area(algebra_visual, "A4", "C6", scale_factor=0.8)

        # 3. Golden Cheetah Visual (#FFD700)
        # Using a stylized representation
        cheetah_icon = Triangle(color="#FFD700").rotate(-PI/2).scale(0.3)
        cheetah_text = Text("Cheetah", font_size=20, color="#FFD700")
        cheetah_visual = VGroup(cheetah_icon, cheetah_text).arrange(RIGHT, buff=0.2)
        # Resolved Issue 32: Moved to E3-F6 to avoid crowding with lecture area
        self.place_in_area(cheetah_visual, "E3", "F6", scale_factor=0.8)

        # Display the collage
        self.play(
            FadeIn(geo_visual),
            FadeIn(algebra_visual),
            FadeIn(cheetah_visual)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Cyan arrows connecting components
        arrow_geo_to_alg = Arrow(
            geo_visual.get_right(), 
            algebra_visual.get_left(), 
            color="#00FFFF", 
            buff=0.3
        )
        
        self.play(Create(arrow_geo_to_alg))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFD700")
        )
        
        # Emphasize real-world bridge
        arrow_alg_to_real = Arrow(
            algebra_visual.get_bottom(), 
            cheetah_visual.get_top(), 
            color="#00FFFF", 
            buff=0.3
        )
        self.play(Create(arrow_alg_to_real))
        self.play(Indicate(cheetah_visual, color="#FFD700"))
        self.wait(2.0)

        # Final transition
        self.play(
            FadeOut(geo_visual),
            FadeOut(algebra_visual),
            FadeOut(cheetah_visual),
            FadeOut(arrow_geo_to_alg),
            FadeOut(arrow_alg_to_real),
            FadeOut(self.lecture),
            FadeOut(self.title)
        )
        
        # Final slogan in white (#FFFFFF)
        final_slogan = Text("Calculus: The Science of Change", font_size=36, color="#FFFFFF")
        # Resolved Issue 31: Constraint final_slogan to avoid obstructing lecture area
        self.place_in_area(final_slogan, "C1", "D6", scale_factor=0.8)
        
        self.play(Write(final_slogan))
        self.wait(2.0)
