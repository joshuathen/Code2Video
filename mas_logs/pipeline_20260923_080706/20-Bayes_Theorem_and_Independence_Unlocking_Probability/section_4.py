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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Detective Robot", [
            "Robots use Bayes to localize themselves.",
            "Sensors provide the new evidence.",
            "Beliefs update to match reality.",
            "A more accurate position is calculated.",
            "Probability is a dynamic estimation process."
        ])
        
        # Grid representation of robot locations
        grid_group = VGroup()
        for pos in self.grid.values():
            dot = Dot(pos, color=WHITE, radius=0.08)
            grid_group.add(dot)
        
        # Load Assets
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        sensor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        
        active_label = Text("Most Likely Position", font_size=16, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_in_area(grid_group, 'A3', 'F6', scale_factor=0.8)
        self.play(FadeIn(grid_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(robot_icon, 'C3', scale_factor=0.5)
        self.place_at_grid(sensor_icon, 'D4', scale_factor=0.4)
        
        sensor_pulse = Circle(radius=0.5, color=RED, stroke_width=2)
        sensor_pulse.move_to(robot_icon.get_center())
        
        self.play(FadeIn(robot_icon), FadeIn(sensor_icon))
        self.play(Create(sensor_pulse), FadeOut(sensor_pulse))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        # Update belief
        self.play(
            *[dot.animate.set_color(GRAY) for dot in grid_group if dot.get_center()[0] < 3.5],
            *[dot.animate.set_color(BLUE) for dot in grid_group if dot.get_center()[0] >= 3.5]
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        target_marker = Dot(self.grid["C3"], color=GOLD, radius=0.12)
        self.play(FadeIn(target_marker))
        self.place_at_grid(active_label, 'C3', scale_factor=0.6)
        active_label.next_to(target_marker, UP, buff=0.1)
        self.play(Write(active_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(BLUE)
        confirm_text = Text("Location Confirmed", font_size=24, color=GREEN)
        self.place_at_grid(confirm_text, 'E3', scale_factor=0.7)
        self.play(Write(confirm_text), Indicate(target_marker, color=YELLOW))
        self.wait(1)
