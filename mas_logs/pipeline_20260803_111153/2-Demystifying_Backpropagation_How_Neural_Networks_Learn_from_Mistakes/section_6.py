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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The network repeats this cycle thousands of times.",
            "We guess, measure error, and backtrack to adjust.",
            "Learning rate determines the size of each update step.",
            "Gradual adjustments lead the network toward perfect mastery.",
            "Iteration transforms random knobs into a smart system."
        ]
        self.setup_layout("The Learning Loop: Iteration to Mastery", lecture_lines)

        # === Animation for Lecture Line 1 ===
        line1_color = "#00FFFF"
        self.lecture[0].set_color(line1_color)
        
        # Display a cyan circular loop (#00FFFF) with arrows indicating a continuous cycle.
        loop = Circle(radius=1.0, color=line1_color)
        arrowhead = Triangle(color=line1_color, fill_opacity=1).scale(0.1).move_to(loop.point_at_angle(0)).rotate(-PI/2)
        loop_graphic = VGroup(loop, arrowhead)
        loop_label = Text("Learning Loop", font_size=24, color=line1_color)
        loop_group = VGroup(loop_graphic, loop_label).arrange(DOWN, buff=0.5)
        self.place_in_area(loop_group, "B2", "E5", scale_factor=1.0)
        
        self.play(Create(loop), FadeIn(arrowhead), Write(loop_label))
        self.play(Rotate(loop_graphic, angle=-2*PI, about_point=loop_graphic.get_center()), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        line2_color = "#FF00FF"
        self.lecture[1].set_color(line2_color)
        
        # Show a purple pulse (#FF00FF) traveling backward through a set of connections.
        layer1 = VGroup(*[Circle(radius=0.15, color=WHITE) for _ in range(3)]).arrange(DOWN, buff=0.4)
        layer2 = VGroup(*[Circle(radius=0.15, color=WHITE) for _ in range(2)]).arrange(DOWN, buff=0.7)
        net_nodes = VGroup(layer1, layer2).arrange(RIGHT, buff=1.2)
        
        connections = VGroup()
        for c1 in layer1:
            for c2 in layer2:
                connections.add(Line(c1.get_right(), c2.get_left(), stroke_width=1.5, color=GRAY))
        
        net_display = VGroup(net_nodes, connections)
        self.place_in_area(net_display, "B2", "E5", scale_factor=1.2)
        
        pulse = Dot(color=line2_color, radius=0.1)
        pulse.move_to(layer2[0].get_center())
        
        self.play(FadeOut(loop_group))
        self.play(Create(net_nodes), Create(connections))
        # Traveling backward from Output to Hidden/Input
        self.play(pulse.animate.move_to(layer1[1].get_center()), run_time=1.5, rate_func=slow_into)
        self.play(FadeOut(pulse))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        line3_color = "#00FF00"
        self.lecture[2].set_color(line3_color)
        
        # Animate a green figure (#00FF00) taking small, controlled steps down a slope.
        slope_line = Line(self.grid["B2"], self.grid["E5"], color=WHITE)
        hiker_dot = Dot(color=line3_color, radius=0.15)
        hiker_dot.move_to(slope_line.get_start())
        
        lr_label = Text("Learning Rate", font_size=20, color=line3_color)
        self.place_at_grid(lr_label, "B5")

        self.play(FadeOut(net_display))
        self.play(Create(slope_line), FadeIn(hiker_dot), Write(lr_label))
        
        # Small steps animation
        step_vector = (slope_line.get_end() - slope_line.get_start()) * 0.18
        for i in range(4):
            self.play(hiker_dot.animate.shift(step_vector), run_time=0.4)
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        line4_color = "#FFFFFF"
        self.lecture[3].set_color(line4_color)
        
        # Show a numerical display 'Error' decreasing from 10.0 to 0.1 (#FFFFFF).
        error_val_tracker = ValueTracker(10.0)
        error_label = Text("Network Error:", font_size=28, color=line4_color)
        error_num_display = DecimalNumber(error_val_tracker.get_value(), num_decimal_places=1, color=line4_color)
        error_display_grp = VGroup(error_label, error_num_display).arrange(DOWN, buff=0.4)
        self.place_in_area(error_display_grp, "C3", "D4", scale_factor=1.2)
        
        error_num_display.add_updater(lambda d: d.set_value(error_val_tracker.get_value()))

        self.play(FadeOut(slope_line), FadeOut(hiker_dot), FadeOut(lr_label))
        self.play(Write(error_display_grp))
        self.play(error_val_tracker.animate.set_value(0.1), run_time=2.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        line5_color = "#FFD700"
        self.lecture[4].set_color(line5_color)
        
        # Highlight the entire network glowing in gold (#FFD700) to represent mastery.
        mastery_nodes1 = VGroup(*[Circle(radius=0.15, color=line5_color, fill_opacity=0.4) for _ in range(3)]).arrange(DOWN, buff=0.4)
        mastery_nodes2 = VGroup(*[Circle(radius=0.15, color=line5_color, fill_opacity=0.4) for _ in range(2)]).arrange(DOWN, buff=0.7)
        mastery_net = VGroup(mastery_nodes1, mastery_nodes2).arrange(RIGHT, buff=1.2)
        
        mastery_conns = VGroup()
        for c1 in mastery_nodes1:
            for c2 in mastery_nodes2:
                mastery_conns.add(Line(c1.get_right(), c2.get_left(), stroke_width=2.5, color=line5_color))
        
        final_system = VGroup(mastery_net, mastery_conns)
        self.place_in_area(final_system, "B2", "E5", scale_factor=1.3)
        
        status_text = Text("Mastery Achieved", font_size=26, color=line5_color)
        self.place_at_grid(status_text, "F3")

        self.play(FadeOut(error_display_grp))
        self.play(Create(final_system), Write(status_text))
        self.play(final_system.animate.scale(1.1), run_time=0.8)
        self.play(final_system.animate.scale(1/1.1), run_time=0.8)
        self.wait(3)
