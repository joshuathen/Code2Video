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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Learning requires repeating this cycle thousands of times.",
            "Every cake baked makes the robot's guesses better.",
            "Soon, Robo-Chef predicts the perfect time every time."
        ]
        self.setup_layout("The Feedback Loop: Mastery through Repetition", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Animate a circular 'Loop' arrow (#00FF00) spinning around the entire network.
        self.lecture[0].set_color(YELLOW)
        
        # Network placeholder nodes and connections
        i1 = Circle(radius=0.15, color=BLUE, fill_opacity=0.8)
        i2 = Circle(radius=0.15, color=BLUE, fill_opacity=0.8)
        h1 = Circle(radius=0.15, color=BLUE, fill_opacity=0.8)
        o1 = Circle(radius=0.15, color=BLUE, fill_opacity=0.8)
        
        input_layer = VGroup(i1, i2).arrange(DOWN, buff=0.3)
        hidden_layer = VGroup(h1).move_to(RIGHT * 0.8)
        output_layer = VGroup(o1).move_to(RIGHT * 1.6)
        
        connections = VGroup(
            Line(i1.get_right(), h1.get_left(), stroke_width=2),
            Line(i2.get_right(), h1.get_left(), stroke_width=2),
            Line(h1.get_right(), o1.get_left(), stroke_width=2)
        ).set_color(GREY)
        
        network = VGroup(input_layer, hidden_layer, output_layer, connections)
        # Fix for Issue 43: Reposition network to avoid cramping
        self.place_in_area(network, 'B3', 'C5', scale_factor=0.8)
        
        # Loop arrow spinning around the network
        loop_circle = Arc(radius=1.2, start_angle=0, angle=TAU*0.8, color="#00FF00")
        loop_circle.add_tip()
        loop_circle.move_to(network.get_center())
        
        self.play(FadeIn(network), Create(loop_circle))
        self.play(Rotate(loop_circle, angle=TAU, about_point=network.get_center(), run_time=2, rate_func=linear))
        
        # === Animation for Lecture Line 2 ===
        # Display an 'Epoch' counter rapidly incrementing from 1 to 1000.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        epoch_label = Text("Epoch:", font_size=24).set_color(WHITE)
        # Fix for Issue 42: Move epoch_label to A3 to avoid crowding
        self.place_at_grid(epoch_label, 'A3', scale_factor=0.8)
        
        epoch_val = ValueTracker(1)
        epoch_num = Integer(1).set_color(WHITE)
        epoch_num.scale(0.8)
        epoch_num.next_to(epoch_label, RIGHT, buff=0.2)
        epoch_num.add_updater(lambda d: d.set_value(int(epoch_val.get_value())))
        
        self.add(epoch_label, epoch_num)
        self.play(Write(epoch_label), FadeIn(epoch_num))
        
        # === Animation for Lecture Line 3 ===
        # Show a line graph where the 'Error' curve drops sharply and flattens near zero.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[0, 1, 0.2],
            x_length=3.5,
            y_length=2.0,
            axis_config={"include_tip": False, "font_size": 18, "color": WHITE}
        )
        
        x_lbl = Text("Epoch", font_size=16).next_to(axes.x_axis, DOWN, buff=0.1)
        y_lbl = Text("Error", font_size=16).rotate(90*DEGREES).next_to(axes.y_axis, LEFT, buff=0.1)
        
        graph_group = VGroup(axes, x_lbl, y_lbl)
        # Fix for Issue 44: Reposition graph_group to avoid clutter
        self.place_in_area(graph_group, 'D3', 'F6', scale_factor=0.9)
        
        # Exponential decay curve for the error
        error_curve = axes.plot(lambda x: 0.9 * np.exp(-x/150), x_range=[0, 1000], color=RED)
        
        self.play(FadeIn(graph_group))
        
        # Simultaneous update: Counter, Curve, and Spinning Loop
        self.play(
            epoch_val.animate.set_value(1000),
            Create(error_curve),
            Rotate(loop_circle, angle=4*TAU, about_point=network.get_center(), run_time=5, rate_func=linear),
            run_time=5
        )
        
        self.wait(2)
