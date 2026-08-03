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
        # Data
        title = "Prerequisite Physics: Conservation Laws"
        lines = [
            "Kinetic energy conservation defines an elliptical velocity relationship.",
            "Momentum conservation governs how velocities change during collisions.",
            "We rescale the velocity axes to simplify the ellipse.",
            "Rescaling transforms the energy ellipse into a perfect circle.",
            "Now, every state becomes a point on this circle."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_ENERGY = WHITE
        COLOR_MOMENTUM = "#87CEEB" # Sky Blue
        COLOR_ELLIPSE = "#FF0000"  # Red
        COLOR_CIRCLE = WHITE
        COLOR_POINT = YELLOW
        
        # Create a TexTemplate that uses PDF output to bypass DVI-to-SVG conversion issues
        custom_tex_template = TexTemplate(output_format=".pdf")
        
        # Pre-build objects for plot to ensure consistent scaling
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY}
        )
        
        # Apply the template to all MathTex instances
        v_label = MathTex("v", color=WHITE, tex_template=custom_tex_template).scale(0.8)
        V_label = MathTex("V", color=WHITE, tex_template=custom_tex_template).scale(0.8)
        
        # Position labels relative to axes
        v_label.move_to(axes.c2p(2.8, 0))
        V_label.move_to(axes.c2p(0, 2.8))
        
        ellipse = Ellipse(width=3, height=1.5, color=COLOR_ELLIPSE)
        circle = Circle(radius=1.5, color=COLOR_CIRCLE)
        
        # Align to axes
        ellipse.move_to(axes.get_center())
        circle.move_to(axes.get_center())
        
        point = Dot(circle.point_at_angle(PI/4), color=COLOR_POINT)
        point_label = MathTex(r"(v', V')", color=COLOR_POINT, tex_template=custom_tex_template).scale(0.7)
        point_label.next_to(point, UR, buff=0.1)
        
        # Group for consistent placement and scaling on right-side grid
        plot_container = VGroup(axes, v_label, V_label, ellipse, circle, point, point_label)
        # Fix Issue 31: Reduce scale factor and maintain placement
        self.place_in_area(plot_container, "C1", "F6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ENERGY)
        
        energy_eq = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color=COLOR_ENERGY, tex_template=custom_tex_template)
        # Fix Issue 29: Use area A2-A5 and scale 0.75 to prevent vertical crowding
        self.place_in_area(energy_eq, "A2", "A5", scale_factor=0.75)
        self.play(Write(energy_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_MOMENTUM)
        
        momentum_eq = MathTex(r"mv_1 + MV_1 = mv_2 + MV_2", color=COLOR_MOMENTUM, tex_template=custom_tex_template)
        # Fix Issue 30: Use area B2-B5 and scale 0.75 to avoid overlap with plot_container
        self.place_in_area(momentum_eq, "B2", "B5", scale_factor=0.75)
        self.play(Write(momentum_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_ELLIPSE)
        
        # Reveal axes and ellipse
        self.play(
            Create(axes), 
            Write(v_label), 
            Write(V_label), 
            Create(ellipse)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_CIRCLE)
        
        # Transform red ellipse into white circle
        self.play(Transform(ellipse, circle))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_POINT)
        
        # Show state point on circle
        self.play(
            Create(point),
            Write(point_label)
        )
        self.wait(3)
