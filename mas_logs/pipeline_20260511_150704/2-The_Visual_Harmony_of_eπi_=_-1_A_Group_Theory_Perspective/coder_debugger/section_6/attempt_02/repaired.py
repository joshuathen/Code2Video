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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Summary: The Bridge Between Algebra and Geometry", 
            [
                "Euler's identity bridges calculus, geometry, and group theory.", 
                "Rotation and growth are two sides of one coin.", 
                "Mathematics finds balance in this single, elegant expression."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color match: Yellow
        self.lecture[0].set_color(YELLOW)
        
        # Create Complex Plane elements
        plane = ComplexPlane(
            x_range=[-2, 2, 1], 
            y_range=[-2, 2, 1], 
            background_line_style={"stroke_opacity": 0.2}
        )
        unit_circle = Circle(radius=1, color=YELLOW, stroke_width=2)
        
        # Key Points
        dot_1 = Dot(plane.n2p(1), color=WHITE)
        dot_i = Dot(plane.n2p(1j), color=WHITE)
        dot_m1 = Dot(plane.n2p(-1), color=WHITE)
        
        # Fixed: Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        label_1 = Text("1", font_size=20).next_to(dot_1, DR, buff=0.1)
        label_i = Text("i", font_size=20, slant=ITALIC).next_to(dot_i, UR, buff=0.1)
        label_m1 = Text("-1", font_size=20).next_to(dot_m1, DL, buff=0.1)
        
        # Visual Anchor for the whole complex plane system
        plane_group = VGroup(plane, unit_circle, dot_1, dot_i, dot_m1, label_1, label_i, label_m1)
        self.place_in_area(plane_group, "A1", "E6", scale_factor=0.8)
        
        # Animation 1: Zoom out effect (scale from small)
        plane_group.save_state()
        plane_group.scale(0.1)
        # Fixed: Changed set_alpha to set_opacity for Manim CE v0.19.0 best practices
        plane_group.set_opacity(0)
        self.play(
            plane_group.animate.scale(10).set_opacity(1),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color match: Teal (Standard Manim replacement for Cyan)
        self.lecture[1].set_color(TEAL)
        
        # Highlights for symmetry: circular path and real axis segment
        p_ref = plane_group[0]
        
        # Rotation path (arc from 1 to -1)
        rotation_path = Arc(
            radius=p_ref.x_axis.get_unit_size(), 
            start_angle=0, 
            angle=PI, 
            color=TEAL, 
            stroke_width=6
        ).move_to(p_ref.n2p(0), aligned_edge=ORIGIN)
        
        # Growth path (linear segment from 1 to -1 on real axis)
        growth_path = Line(
            p_ref.n2p(1), 
            p_ref.n2p(-1), 
            color=TEAL, 
            stroke_width=6
        ).set_stroke(opacity=0.6)
        
        self.play(
            Create(rotation_path),
            Create(growth_path),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color match: Gold (#FFD700)
        self.lecture[2].set_color("#FFD700")
        
        # Unified Formula - Fixed: Replaced MathTex with Text
        formula = Text("e^πi + 1 = 0", color="#FFD700", font_size=42)
        # Position at the bottom of the right side grid (Row F)
        self.place_in_area(formula, "F1", "F6", scale_factor=1.0)
        
        self.play(FadeIn(formula, shift=UP), run_time=1.5)
        self.play(Indicate(formula, color="#FFD700", scale_factor=1.1), run_time=1.5)
        
        self.wait(3)