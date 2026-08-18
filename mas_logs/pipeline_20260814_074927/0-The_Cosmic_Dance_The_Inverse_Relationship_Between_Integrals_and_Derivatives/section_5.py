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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        self.setup_layout("Inverse Operations: The Mathematical U-Turn", [
            "Differentiation and integration are inverse mathematical operations.",
            "Differentiate position to find the velocity at any moment.",
            "Integrate velocity to recover the original position function.",
            "They undo each other like addition and subtraction.",
            "This symmetry is the heart of the calculus loop."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Two boxes labeled 'Position' (#FFD700) and 'Velocity' (#ADFF2F) appear.
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        pos_rect = Rectangle(width=2.2, height=1.0, color="#FFD700")
        pos_text = Text("Position", font_size=24, color="#FFD700")
        pos_box = VGroup(pos_rect, pos_text)
        self.place_at_grid(pos_box, "B2", scale_factor=0.8)
        
        vel_rect = Rectangle(width=2.2, height=1.0, color="#ADFF2F")
        vel_text = Text("Velocity", font_size=24, color="#ADFF2F")
        vel_box = VGroup(vel_rect, vel_text)
        self.place_at_grid(vel_box, "B5", scale_factor=0.8)
        
        self.play(FadeIn(pos_box), FadeIn(vel_box))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A top arrow 'd/dx' (#FF4500) moves from Position to Velocity.
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        
        deriv_arrow = CurvedArrow(
            start_point=pos_box.get_right() + UP * 0.2,
            end_point=vel_box.get_left() + UP * 0.2,
            angle=-PI/4,
            color="#FF4500"
        )
        deriv_label = MathTex(r"\frac{d}{dx}", color="#FF4500", font_size=36)
        self.place_in_area(deriv_label, "A3", "A4", scale_factor=0.8)
        
        self.play(Create(deriv_arrow), Write(deriv_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A bottom curved arrow '∫' (#7CFC00) moves from Velocity to Position.
        self.play(self.lecture[2].animate.set_color("#7CFC00"))
        
        int_arrow = CurvedArrow(
            start_point=vel_box.get_left() + DOWN * 0.2,
            end_point=pos_box.get_right() + DOWN * 0.2,
            angle=-PI/4,
            color="#7CFC00"
        )
        int_label = MathTex(r"\int", color="#7CFC00", font_size=48)
        self.place_in_area(int_label, "C3", "C4", scale_factor=0.8)
        
        self.play(Create(int_arrow), Write(int_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Both boxes glow white (#FFFFFF) simultaneously to show the link.
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        glow_pos = pos_rect.copy().set_stroke(color=WHITE, width=10).set_fill(WHITE, opacity=0.4)
        glow_vel = vel_rect.copy().set_stroke(color=WHITE, width=10).set_fill(WHITE, opacity=0.4)
        
        self.play(
            FadeIn(glow_pos),
            FadeIn(glow_vel),
            run_time=0.5
        )
        self.play(
            FadeOut(glow_pos),
            FadeOut(glow_vel),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Large text 'Mathematical U-Turn' appears in pink (#FF69B4).
        self.play(self.lecture[4].animate.set_color("#FF69B4"))
        
        u_turn_text = Text("Mathematical U-Turn", color="#FF69B4", font_size=42)
        self.place_in_area(u_turn_text, "E2", "F5")
        
        self.play(Write(u_turn_text))
        self.wait(2)
