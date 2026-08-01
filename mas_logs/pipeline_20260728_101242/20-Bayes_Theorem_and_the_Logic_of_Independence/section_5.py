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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The 'Sick Robot' Case Study"
        lines = [
            "Ten robots out of a thousand are broken.",
            "Our sensor is ninety percent accurate for failures.",
            "The sensor beeps: is the robot actually broken?",
            "We must account for the rare failure rate.",
            "Bayes reveals the true probability behind the alert."
        ]
        self.setup_layout(title, lines)

        # Assets
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        sensor_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg"

        # === Animation for Lecture Line 1 ===
        # Display a 10x10 grid of robot icons [Asset: robot.svg] (#888888) to represent the population.
        self.lecture[0].set_color("#888888")
        
        # Create a 10x10 grid of robot icons
        robots = VGroup(*[SVGMobject(robot_path).set_color("#888888").scale(0.15) for _ in range(100)])
        robots.arrange_in_grid(rows=10, cols=10, buff=0.15)
        self.place_in_area(robots, "A1", "F6", scale_factor=0.9)
        
        self.play(FadeIn(robots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Turn 1 robot icon [Asset: robot.svg] bright red (#FF0000) to signify the "Broken" prior.
        self.lecture[1].set_color("#FF0000")
        
        broken_robot = robots[0]
        self.play(broken_robot.animate.set_color("#FF0000").scale(1.5))
        self.play(broken_robot.animate.scale(1/1.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a sensor [Asset: sensor.svg] emitting a yellow beam (#FFFF00) 
        # scanning a healthy robot [Asset: robot.svg] and triggering a "Beep".
        self.lecture[2].set_color("#FFFF00")
        
        # Target a healthy robot (index 55)
        target_robot = robots[55]
        
        sensor = SVGMobject(sensor_path).set_color("#FFFF00").scale(0.4)
        sensor.move_to(target_robot.get_center() + UP * 0.8)
        
        beam = Triangle(color="#FFFF00", fill_opacity=0.3, stroke_width=0).scale(0.3).rotate(PI)
        beam.next_to(sensor, DOWN, buff=0)
        
        beep_label = Text("BEEP!", font_size=20, color="#FFFF00")
        # Issue 41: Position beep_label at A6
        self.place_at_grid(beep_label, "A6", scale_factor=0.8)
        
        self.play(FadeIn(sensor))
        self.play(Create(beam))
        self.play(FadeIn(beep_label))
        self.play(target_robot.animate.set_color("#FFFF00"))
        self.wait(1)
        self.play(FadeOut(sensor), FadeOut(beam), FadeOut(beep_label))

        # === Animation for Lecture Line 4 ===
        # Pop up a calculation window showing the ratio of true positives to total beeps.
        self.lecture[3].set_color(WHITE)
        
        calc_box = RoundedRectangle(corner_radius=0.1, width=4.5, height=3, color=WHITE, fill_color=BLACK, fill_opacity=0.9)
        # Issue 43: Change scale_factor to 0.8
        self.place_in_area(calc_box, "B1", "E6", scale_factor=0.8)
        
        # Bayes calculation: P(Broken|Beep) approx 8.3%
        calc_text = MathTex(
            r"P(\text{Broken} \mid \text{Beep}) = \frac{P(\text{Beep} \mid \text{Broken}) \cdot P(\text{Broken})}{P(\text{Beep})}",
            r"= \frac{0.9 \cdot 0.01}{0.108} \approx 8.3\%",
            font_size=24, color=WHITE
        ).arrange(DOWN)
        calc_text.move_to(calc_box.get_center())
        
        self.play(Create(calc_box))
        self.play(Write(calc_text))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Display the conclusion "Still Likely Healthy" in bold green text (#00FF00).
        self.lecture[4].set_color("#00FF00")
        
        conclusion = Text("STILL LIKELY HEALTHY", font_size=36, color="#00FF00", weight=BOLD)
        # Issue 42: Position conclusion in area F1 to F6
        self.place_in_area(conclusion, "F1", "F6", scale_factor=0.8)
        
        self.play(Write(conclusion))
        self.wait(3)
