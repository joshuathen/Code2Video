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
        self.setup_layout(
            "Application: The Curse of Dimensionality", 
            [
                "High-dimensional data creates challenges for AI and search.",
                "Points become sparse and distances lose their meaning.",
                "Every data point feels equally far from all others."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show a query point surrounded by data points. Highlight the 'nearest' neighbor.
        self.lecture[0].set_color(BLUE_D)
        
        # Issue 36: Move query point to C4 to avoid overlap with lecture
        query_point = Dot(color=WHITE)
        self.place_at_grid(query_point, "C4")
        query_label = Text("Query", font_size=16, color=WHITE)
        query_label.next_to(query_point, UP, buff=0.1)

        data_points = VGroup(
            Dot(color=BLUE).move_to(self.grid["B3"]),
            Dot(color=BLUE).move_to(self.grid["B5"]),
            Dot(color=BLUE).move_to(self.grid["D3"]),
            Dot(color=BLUE).move_to(self.grid["D5"]),
            Dot(color=BLUE).move_to(self.grid["C5"]) # Nearest one
        )
        
        self.play(FadeIn(query_point), FadeIn(query_label))
        self.play(Create(data_points))
        
        # Highlight nearest neighbor
        nearest_line = Line(query_point.get_center(), data_points[4].get_center(), color=GREEN)
        self.play(Create(nearest_line))
        self.play(data_points[4].animate.set_color(GREEN).scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the points moving apart. All distance arrows turn #FFFF00 and equal length.
        self.lecture[1].set_color(YELLOW)
        
        # Positions scattered to edges to show sparsity
        target_positions = ["A3", "A6", "F3", "F6", "D6"]
        
        distance_lines = VGroup()
        for i in range(len(data_points)):
            line = Line(query_point.get_center(), data_points[i].get_center(), color=BLUE_B, stroke_width=2)
            distance_lines.add(line)
        
        self.play(FadeOut(nearest_line), FadeIn(distance_lines))
        
        # Move points and update lines
        animations = []
        for i, point in enumerate(data_points):
            # Use updater to keep lines connected
            line = distance_lines[i]
            line.add_updater(lambda l, p=point, q=query_point: l.put_start_and_end_on(q.get_center(), p.get_center()))
            animations.append(point.animate.move_to(self.grid[target_positions[i]]))
            
        self.play(*animations, run_time=2)
        self.wait(0.5)
        
        # Change lines to yellow and "equal" length appearance
        for line in distance_lines:
            line.clear_updaters()
            
        self.play(
            distance_lines.animate.set_color("#FFFF00"),
            data_points.animate.set_color("#FFFF00"),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Robot icon #FFFFFF appears, looking confused.
        self.lecture[2].set_color(WHITE)
        
        # Issue 24: Use Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg
        # Issue 37: Place at F5
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot_icon.set_color(WHITE)
        self.place_at_grid(robot_icon, "F5", scale_factor=0.8)
        
        # Robot's confusion marker
        q_mark = Text("?", color=WHITE, font_size=36).next_to(robot_icon, UP, buff=0.2)
        
        # All distances look equal
        labels = VGroup()
        for i, point in enumerate(data_points):
            dist_label = MathTex(r"d \approx \infty", font_size=20, color="#FFFF00")
            # Anchor label one unit away (B012) or at midpoint
            midpoint = Line(query_point.get_center(), point.get_center()).get_center()
            dist_label.move_to(midpoint + UP*0.3)
            labels.add(dist_label)

        self.play(FadeIn(robot_icon))
        self.play(Write(q_mark))
        self.play(FadeIn(labels))
        
        # Final emphasis on sparsity
        self.wait(2)
