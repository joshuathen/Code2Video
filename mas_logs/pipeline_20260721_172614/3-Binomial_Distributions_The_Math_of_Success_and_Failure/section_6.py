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
            "Binomial distributions help predict real-world manufacturing defects.",
            "They allow factories to plan for expected failure rates.",
            "Remember the BINS criteria for any binomial problem."
        ]
        self.setup_layout("Real-World Application & Summary", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a conveyor belt with a robot scanning items; highlight 'Pass' (#00FF00) and 'Fail' (#FF0000).
        self.lecture[0].set_color(YELLOW)
        
        conveyor = Line(self.grid["E1"] + LEFT*0.5, self.grid["E6"] + RIGHT*0.5, color=GREY)
        robot_body = Square(side_length=0.8, color=BLUE_B, fill_opacity=1)
        robot_eye_l = Circle(radius=0.1, color=WHITE, fill_opacity=1).shift(UP*0.15 + LEFT*0.15)
        robot_eye_r = Circle(radius=0.1, color=WHITE, fill_opacity=1).shift(UP*0.15 + RIGHT*0.15)
        robot = VGroup(robot_body, robot_eye_l, robot_eye_r)
        self.place_at_grid(robot, "C3", scale_factor=0.8)
        
        # Scanning beam
        beam = Polygon(self.grid["C3"], self.grid["E3"] + LEFT*0.25, self.grid["E3"] + RIGHT*0.25, 
                       color=YELLOW, fill_opacity=0.3, stroke_width=0)
        
        self.add(conveyor, robot)
        
        results = ["#00FF00", "#FF0000", "#00FF00"]
        labels = ["Pass", "Fail", "Pass"]
        
        items = []
        for i in range(3):
            item = Square(side_length=0.4, fill_opacity=1, color=GREY_B)
            items.append(item)
            item.move_to(self.grid["E1"] + LEFT*1.5)
            
            # Slide to scanning station
            self.play(item.animate.move_to(self.grid["E3"]), run_time=0.6, rate_func=linear)
            
            # Scan animation
            self.add(beam)
            self.play(item.animate.set_color(results[i]), run_time=0.2)
            
            res_text = Text(labels[i], font_size=18, color=results[i])
            self.place_at_grid(res_text, "D3")
            self.add(res_text)
            
            self.play(Flash(item, color=results[i], flash_radius=0.3, run_time=0.3))
            self.remove(beam)
            
            # Slide off and fade label
            self.play(
                item.animate.move_to(self.grid["E6"] + RIGHT*1.5),
                res_text.animate.shift(UP*0.4).set_opacity(0),
                run_time=0.6,
                rate_func=linear
            )
            self.remove(res_text)

        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Display a digital factory report showing a predicted defect rate calculation.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Clear items/robot
        self.play(FadeOut(robot, conveyor, *items))
        
        report_frame = Rectangle(width=4.5, height=3.5, color=BLUE_D, fill_opacity=0.1)
        self.place_in_area(report_frame, "B1", "F6")
        
        report_title = Text("FACTORY DEFECT ANALYSIS", font_size=22, color=BLUE_A)
        # Fix for Issue 36: Move title to row A to avoid overlap with frame border at row B
        self.place_at_grid(report_title, "A3")
        
        calc_content = VGroup(
            MathTex("n = 100\\text{ items tested}", font_size=30),
            MathTex("p = 0.02\\text{ defect probability}", font_size=30),
            MathTex("E[X] = n \\cdot p = 2\\text{ expected defects}", font_size=32, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.place_in_area(calc_content, "C2", "E5")
        
        self.play(Create(report_frame), Write(report_title))
        self.play(Write(calc_content))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Flash the BINS acronym one last time in a golden color (#FFD700).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(report_frame, report_title, calc_content))
        
        bins_text = Text("BINS", font_size=80, color="#FFD700", weight=BOLD)
        # Fix for Issue 37: Reduce space occupied by bins_text to avoid dominance
        self.place_in_area(bins_text, "B2", "C5")
        
        criteria_list = VGroup(
            Text("B: Binary outcomes (S/F)", font_size=26, color="#FFD700"),
            Text("I: Independent trials", font_size=26, color="#FFD700"),
            Text("N: Number of trials fixed", font_size=26, color="#FFD700"),
            Text("S: Same prob. of success", font_size=26, color="#FFD700")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        # Fix for Issue 38: Use more space for criteria_list to avoid cramped look
        self.place_in_area(criteria_list, "D2", "F5")
        
        self.play(Write(bins_text))
        self.play(Flash(bins_text, color="#FFD700", flash_radius=1.5, num_lines=12))
        self.play(
            bins_text.animate.scale(0.5).move_to(self.grid["B3"]),
            FadeIn(criteria_list, shift=UP)
        )
        
        self.wait(3)
