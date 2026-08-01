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
        # Setup the layout with section details from the storyboard
        self.setup_layout("The Brute Force Challenge", [
            "Brute force attempts every possible combination to crack.",
            "Even billion-speed supercomputers cannot break it quickly.",
            "It would take trillions of years to finish."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Description: A white robot icon (#FFFFFF).
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        
        robot_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").set_color(WHITE)
        self.place_at_grid(robot_asset, "B2", scale_factor=0.8)
        
        self.play(FadeIn(robot_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: A fast '1 Quadrillion H/s' counter (#ADD8E6).
        # Line color: Light Blue (#ADD8E6)
        
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        
        # Creating the counter display
        counter_val = Integer(0, color="#ADD8E6")
        counter_unit = Text(" Quadrillion H/s", font_size=18, color="#ADD8E6")
        counter_group = VGroup(counter_val, counter_unit).arrange(RIGHT, buff=0.1)
        
        # Place it near the robot
        self.place_in_area(counter_group, "B3", "B5", scale_factor=0.9)
        
        self.play(FadeIn(counter_group))
        # Animate the counter value to represent massive speed
        self.play(ChangeDecimalToValue(counter_val, 1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: A golden timer (#FFD700) counts up and a red progress bar (#FF0000).
        # Line color: Gold (#FFD700)
        
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Timer at D2
        timer_circle = Circle(radius=0.4, color="#FFD700")
        timer_hand = Line(timer_circle.get_center(), timer_circle.get_top(), color="#FFD700")
        timer_label = Text("Trillions of Years", font_size=20, color="#FFD700")
        timer_vgroup = VGroup(timer_circle, timer_hand, timer_label).arrange(DOWN, buff=0.2)
        
        self.place_at_grid(timer_vgroup, "D2", scale_factor=0.8)
        
        # Progress Bar at E2-F5
        progress_bg = Rectangle(width=4.0, height=0.5, color=WHITE).set_stroke(width=2)
        # Red fill starts very small to emphasize the futility
        progress_fill = Rectangle(width=0.01, height=0.4, color="#FF0000", fill_opacity=0.9).align_to(progress_bg, LEFT).shift(RIGHT*0.05)
        progress_text = Text("0.000...01% Complete", font_size=18, color="#FF0000").next_to(progress_bg, UP, buff=0.2)
        progress_group = VGroup(progress_bg, progress_fill, progress_text)
        
        # Placing in a slightly larger area to ensure separation from the timer
        self.place_in_area(progress_group, "E2", "F5", scale_factor=0.9)
        
        self.play(FadeIn(timer_vgroup), FadeIn(progress_group))
        
        # Rotate timer hand to simulate time passing rapidly
        self.play(
            Rotate(timer_hand, angle=-10*PI, about_point=timer_circle.get_center()), 
            run_time=4, 
            rate_func=linear
        )
        
        self.wait(2)
