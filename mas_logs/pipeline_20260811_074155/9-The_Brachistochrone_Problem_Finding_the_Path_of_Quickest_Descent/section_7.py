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
        # Define color constants
        CYAN = "#00FFFF"
        YELLOW_COL = "#FFFF00"
        
        # Fetching storyboard data
        title_text = "Summary and Real-World Impact"
        lecture_lines = [
            "We've seen that the fastest path isn't always straight.",
            "This problem pioneered the powerful calculus of variations.",
            "These principles now optimize modern aerospace and robotics."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Set initial lecture color to GRAY for highlighting
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1: We've seen that the fastest path isn't always straight.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Summary text: 'Shortest path isn\'t always fastest' (#FFFFFF)
        # Corrected positioning to A2-B5 (Issue 34)
        summary_text = Text("Shortest path isn't always fastest", font_size=24, color=WHITE)
        self.place_in_area(summary_text, 'A2', 'B5')
        self.play(Write(summary_text))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2: This problem pioneered the powerful calculus of variations.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(YELLOW_COL)
        )
        
        # Animation 2: Display the text 'Calculus of Variations' (#FFFF00)
        # Corrected positioning to C2-D5 (Issue 35)
        calc_var_text = Text("Calculus of Variations", font_size=30, color=YELLOW_COL)
        self.place_in_area(calc_var_text, 'C2', 'D5')
        self.play(FadeIn(calc_var_text))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3: These principles now optimize modern aerospace and robotics.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(CYAN)
        )
        
        # Animation 3: Show simple icons representing aerospace and robotics.
        # Integrate Asset: robot.svg (Issue 24)
        try:
            robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
            robot_icon.set_color(CYAN)
        except Exception:
            # Fallback for robot icon if asset path is invalid
            robot_base = Circle(radius=0.1, color=CYAN, fill_opacity=0.8)
            robot_arm_seg = Rectangle(width=0.08, height=0.4, color=CYAN, fill_opacity=0.8).next_to(robot_base, UP, buff=0).rotate(-PI/4, about_point=robot_base.get_center())
            robot_icon = VGroup(robot_base, robot_arm_seg)
        
        # Aerospace Icon (Rocket)
        rocket_body = Rectangle(width=0.2, height=0.4, color=CYAN, fill_opacity=0.8)
        rocket_nose = Triangle(color=CYAN, fill_opacity=0.8).scale(0.15).next_to(rocket_body, UP, buff=0)
        rocket = VGroup(rocket_body, rocket_nose)
        
        # Corrected positioning to E2, E5 with scale_factor=1.2 (Issue 36)
        self.place_at_grid(rocket, 'E2', scale_factor=1.2)
        self.place_at_grid(robot_icon, 'E5', scale_factor=1.2)
        
        self.play(FadeIn(rocket), FadeIn(robot_icon))
        self.wait(2)

        # === Final Closure for Animation 3 ===
        # "Fade out everything to the summary text: 'Shortest path isn\'t always fastest'"
        self.play(
            FadeOut(calc_var_text),
            FadeOut(rocket),
            FadeOut(robot_icon),
            self.lecture[2].animate.set_color(GRAY)
        )
        # Final focus on the summary text
        self.play(summary_text.animate.scale(1.25))
        self.wait(3)
