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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Changing the Perspective: Velocity Space", 
            [
                "Let's map the velocities on a coordinate grid.",
                "Rescaling the axes by mass simplifies the math.",
                "The energy ellipse transforms into a perfect circle."
            ]
        )
        
        # Colors
        YELLOW_DOT = "#FFFF00"
        RED_ELLIPSE = "#FF4444"
        CYAN_CIRCLE = "#00FFFF"
        AXIS_COLOR = "#D3D3D3"
        LABEL_COLOR = "#FFFFFF"

        # Initialize axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": AXIS_COLOR, "include_tip": True},
            tips=False
        )
        
        # Labels for the axes
        labels = axes.get_axis_labels(
            x_label=MathTex("v_1", font_size=24, color=LABEL_COLOR), 
            y_label=MathTex("v_2", font_size=24, color=LABEL_COLOR)
        )

        # The energy constraint: 1/2 m v1^2 + 1/2 M v2^2 = E
        # Stretched horizontally assuming m < M
        ellipse = Ellipse(width=4.8, height=2.2, color=RED_ELLIPSE)
        
        # Dot representing velocities
        start_angle = 35 * DEGREES
        v_dot = Dot(point=ellipse.point_at_angle(start_angle), color=YELLOW_DOT, radius=0.1)
        dot_label = MathTex("(v_1, v_2)", font_size=24, color=YELLOW_DOT)
        dot_label.next_to(v_dot, UR, buff=0.15)

        # Container for the coordinate system to use the grid system
        visual_group = VGroup(axes, labels, ellipse, v_dot, dot_label)
        
        # Fix for Issues 29 & 30: Use area B2 to F5 and scale factor 0.7 
        # to ensure the assembly is inward from edges and below the title.
        self.place_in_area(visual_group, "B2", "F5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Let's map the velocities on a coordinate grid.
        self.lecture[0].set_color(YELLOW_DOT)
        self.play(
            Create(axes),
            Write(labels),
            run_time=1.5
        )
        self.play(
            Create(ellipse),
            FadeIn(v_dot),
            FadeIn(dot_label),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Rescaling the axes by mass simplifies the math.
        self.lecture[1].set_color(WHITE)
        
        # Transition: Scaled labels sqrt(m)*v1 and sqrt(M)*v2
        scaled_labels = axes.get_axis_labels(
            x_label=MathTex("\\sqrt{m}v_1", font_size=24, color=LABEL_COLOR), 
            y_label=MathTex("\\sqrt{M}v_2", font_size=24, color=LABEL_COLOR)
        )

        self.play(
            Transform(labels, scaled_labels),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The energy ellipse transforms into a perfect circle.
        self.lecture[2].set_color(CYAN_CIRCLE)
        
        # Ellipse becomes a circle in rescaled space
        circle_radius = 2.0
        circle = Circle(radius=circle_radius, color=CYAN_CIRCLE).move_to(axes.get_center())
        
        # Update dot position on the circle
        target_pos = circle.point_at_angle(start_angle)
        
        self.play(
            Transform(ellipse, circle),
            v_dot.animate.move_to(target_pos),
            dot_label.animate.next_to(target_pos, UR, buff=0.15),
            run_time=2
        )
        self.wait(1)

        # Jumps on the circle boundary representing the effect of collisions
        jump_angles = [135 * DEGREES, 225 * DEGREES, 315 * DEGREES]
        
        for angle in jump_angles:
            new_pos = circle.point_at_angle(angle)
            # Create a curved path for the dot's "jump"
            jump_path = ArcBetweenPoints(v_dot.get_center(), new_pos, angle=-TAU/6)
            
            # Position label strategically based on destination quadrant
            if angle < PI: 
                buff_dir = UL
            elif angle < 1.5 * PI:
                buff_dir = DL
            else:
                buff_dir = DR
                
            self.play(
                MoveAlongPath(v_dot, jump_path),
                dot_label.animate.next_to(new_pos, buff_dir, buff=0.15),
                run_time=1.2
            )
            self.wait(0.5)

        self.wait(3)
