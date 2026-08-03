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

class Section4Scene(TeachingScene):
    def construct(self):
        title = "Mapping Physics to Geometry (Phase Space)"
        lines = [
            "Scale the velocities to transform physics into geometry.",
            "The conservation of energy now forms a perfect circle.",
            "Each collision jumps to a new point on the circle.",
            "Every impact represents a fixed angle of rotation.",
            "Physics becomes a journey around a circular path."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_ELLIPSE = YELLOW
        COLOR_CIRCLE = "#00FFFF"
        COLOR_POINT = "#FFFFFF"
        COLOR_CHORD = "#FF0000"

        # Coordinates/Axes Setup
        # Fix: Using 'B2' to 'F5' as requested in Issue 28 for better visibility
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        self.place_in_area(axes, "B2", "F5")
        
        v_label = Text("v", font_size=18).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        V_label = Text("V", font_size=18).next_to(axes.y_axis.get_end(), UP, buff=0.1)
        plot_group = VGroup(axes, v_label, V_label)

        # === Animation for Lecture Line 1 ===
        # Scale the velocities to transform physics into geometry.
        self.lecture[0].set_color(COLOR_ELLIPSE)
        
        # Initial Ellipse representing conservation of energy: m v^2 + M V^2 = const
        # Visually show a flattened ellipse to represent different masses
        ellipse = Ellipse(
            width=axes.x_axis.get_unit_size() * 4, 
            height=axes.y_axis.get_unit_size() * 1.2, 
            color=COLOR_ELLIPSE
        )
        ellipse.move_to(axes.c2p(0, 0))
        
        self.play(FadeIn(plot_group), Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The conservation of energy now forms a perfect circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_CIRCLE)
        
        # Transform ellipse to circle by scaling the y-axis
        # Circle radius in pixels mapped to 2 units on axes
        circle_radius_px = axes.x_axis.get_unit_size() * 2
        circle = Circle(radius=circle_radius_px, color=COLOR_CIRCLE)
        circle.move_to(axes.c2p(0, 0))
        
        # Scaled label for the V axis indicating the transform V' = V * sqrt(M/m)
        V_scaled_label = MathTex(r"V \sqrt{M/m}", font_size=18).move_to(V_label.get_center())
        
        self.play(
            ReplacementTransform(ellipse, circle),
            ReplacementTransform(V_label, V_scaled_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Each collision jumps to a new point on the circle.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_POINT)
        
        # Initial state point on the circle
        start_angle = 0 * DEGREES
        point_start = Dot(axes.c2p(2 * np.cos(start_angle), 2 * np.sin(start_angle)), color=COLOR_POINT, radius=0.08)
        
        self.play(FadeIn(point_start))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Every impact represents a fixed angle of rotation.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_CHORD)
        
        # A collision is a jump. The angle depends on mass ratio.
        # Fixed angle theta for visualization of the chord.
        theta = 40 * DEGREES 
        
        # First jump: from (v, V) to new state
        pos1 = axes.c2p(2 * np.cos(start_angle + theta), 2 * np.sin(start_angle + theta))
        point_1 = Dot(pos1, color=COLOR_POINT, radius=0.08)
        chord_1 = Line(point_start.get_center(), point_1.get_center(), color=COLOR_CHORD)
        
        self.play(Create(chord_1), FadeIn(point_1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Physics becomes a journey around a circular path.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_CIRCLE)
        
        # Draw a sequence of chords representing successive collisions
        last_point = point_1
        curr_angle = start_angle + theta
        for _ in range(5):
            next_angle = curr_angle + theta
            new_pos = axes.c2p(2 * np.cos(next_angle), 2 * np.sin(next_angle))
            new_dot = Dot(new_pos, color=COLOR_POINT, radius=0.08)
            new_chord = Line(last_point.get_center(), new_dot.get_center(), color=COLOR_CHORD)
            
            self.play(Create(new_chord), FadeIn(new_dot), run_time=0.4)
            last_point = new_dot
            curr_angle = next_angle

        self.wait(2)

        # Final cleanup for color reset
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
