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
        title_text = "Prerequisite Knowledge: Conservation Laws as Geometry"
        lecture_lines = [
            "Conservation of energy creates an elliptical state space.",
            "Rescaling the velocity axes transforms this into a circle.",
            "Every state of the system lies on this circle.",
            "Collisions move the system's state along the circle.",
            "Physics is now transformed into a geometric problem."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display the energy conservation equation 1/2 Mv² + 1/2 mu² = E in white (#FFFFFF).
        # A yellow (#FADB14) ellipse is plotted on a graph with v and u axes.
        self.lecture[0].set_color("#FADB14")
        
        equation = MathTex(r"\frac{1}{2} M v^2 + \frac{1}{2} m u^2 = E", color=WHITE)
        # Fixed: Updated grid placement for better centering as per Issue 30
        self.place_in_area(equation, "A2", "A5", scale_factor=0.8)
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=True
        )
        v_label = MathTex("v", color=WHITE).scale(0.8)
        u_label = MathTex("u", color=WHITE).scale(0.8)
        
        # Positioning labels within grid constraints
        v_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.2)
        u_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.2)
        
        graph_elements = VGroup(axes, v_label, u_label)
        # Fixed: Updated grid area and scale to prevent labels from hitting the edge as per Issue 31
        self.place_in_area(graph_elements, "B2", "F5", scale_factor=0.85)
        
        # Initial Ellipse
        # Semi-major axis 2.2, semi-minor axis 1.2 in axis units
        ellipse = Ellipse(
            width=2.2 * axes.x_axis.get_unit_size() * 2,
            height=1.2 * axes.y_axis.get_unit_size() * 2,
            color="#FADB14"
        )
        ellipse.move_to(axes.c2p(0, 0))
        
        self.play(Write(equation))
        self.play(Create(axes), Write(v_label), Write(u_label))
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The v-axis scales by sqrt(M), transforming the yellow ellipse into a perfect circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FADB14")
        
        # Target circle
        circle_radius = 1.2 * axes.y_axis.get_unit_size()
        circle = Circle(radius=circle_radius, color="#FADB14")
        circle.move_to(axes.c2p(0, 0))
        
        # Updated label for rescaled axis
        v_scaled_label = MathTex(r"\sqrt{M}v", color=WHITE).scale(0.8)
        v_scaled_label.move_to(v_label.get_center())

        self.play(
            Transform(ellipse, circle),
            Transform(v_label, v_scaled_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Every state of the system lies on this circle."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FADB14")
        
        state_point = Dot(color=BLUE)
        # Position at 30 degrees (pi/6)
        state_point.move_to(circle.point_from_proportion(0.1))
        state_label = Text("State", font_size=18, color=BLUE)
        state_label.next_to(state_point, UR, buff=0.1)
        
        self.play(FadeIn(state_point), Write(state_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Collisions move the system's state along the circle."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FADB14")
        
        # Use ValueTracker for efficient movement along circle
        angle_tracker = ValueTracker(0.1)
        
        def update_point(mob):
            mob.move_to(circle.point_from_proportion(angle_tracker.get_value() % 1.0))
            
        def update_label(mob):
            mob.next_to(state_point, UR, buff=0.1)
            
        state_point.add_updater(update_point)
        state_label.add_updater(update_label)
        
        self.play(angle_tracker.animate.set_value(0.4), run_time=2, rate_func=linear)
        self.wait(0.5)
        
        state_point.clear_updaters()
        state_label.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # "Physics is now transformed into a geometric problem."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FADB14")
        
        self.play(Indicate(circle), Flash(state_point, color=BLUE))
        self.wait(2)
