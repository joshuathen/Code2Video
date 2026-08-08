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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Application: The 'Broken Robot' Test",
            [
                "A robot scans chips for rare defects.",
                "Only 1% of chips are actually defective.",
                "The robot is 99% accurate but has false alarms.",
                "If it beeps, is the chip definitely broken?",
                "Surprisingly, most beeps come from healthy chips!"
            ]
        )

        # Colors
        line_colors = [WHITE, "#FFFF00", "#00FF00", "#00FFFF", "#FF0000"]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_colors[0]))
        
        root_text = Text("1,000 Chips", font_size=20, color=WHITE)
        chip_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/chip.svg").set_color(WHITE)
        
        root_group = VGroup(chip_icon, root_text).arrange(DOWN, buff=0.1)
        self.place_in_area(root_group, "A3", "A4", scale_factor=0.6)
        
        self.play(FadeIn(root_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(line_colors[1]))
        
        def_text = Text("10 Defective (1%)", font_size=18, color="#FFFF00")
        hlth_text = Text("990 Healthy (99%)", font_size=18, color=WHITE)
        
        # Grid positions adjusted per Issue 43
        self.place_at_grid(def_text, "B2", scale_factor=0.8)
        self.place_at_grid(hlth_text, "B5", scale_factor=0.8)
        
        line_l = Line(self.grid["A3.5"] if "A3.5" in self.grid else (self.grid["A3"]+self.grid["A4"])/2, 
                     self.grid["B2"], buff=0.3, color=GRAY)
        line_r = Line(self.grid["A3.5"] if "A3.5" in self.grid else (self.grid["A3"]+self.grid["A4"])/2, 
                     self.grid["B5"], buff=0.3, color=GRAY)
        
        self.play(Create(line_l), Create(line_r))
        self.play(FadeIn(def_text), FadeIn(hlth_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(line_colors[2]))
        
        # From Defective: 99% accurate -> 9.9 Beep
        beep_def = Text("9.9 Beep", font_size=16, color="#00FF00")
        no_beep_def = Text("0.1 No Beep", font_size=16, color=GRAY)
        
        # From Healthy: 5% false alarm -> 49.5 Beep
        beep_hlth = Text("49.5 Beep", font_size=16, color="#00FF00")
        no_beep_hlth = Text("940.5 No Beep", font_size=16, color=GRAY)
        
        # Grid positions adjusted per Issue 43
        self.place_at_grid(beep_def, "C1", scale_factor=0.8)
        self.place_at_grid(no_beep_def, "C3", scale_factor=0.8)
        self.place_at_grid(beep_hlth, "C4", scale_factor=0.8)
        self.place_at_grid(no_beep_hlth, "C6", scale_factor=0.8)
        
        l1 = Line(self.grid["B2"], self.grid["C1"], buff=0.2, color=GRAY_A)
        l2 = Line(self.grid["B2"], self.grid["C3"], buff=0.2, color=GRAY_A)
        l3 = Line(self.grid["B5"], self.grid["C4"], buff=0.2, color=GRAY_A)
        l4 = Line(self.grid["B5"], self.grid["C6"], buff=0.2, color=GRAY_A)
        
        self.play(Create(l1), Create(l2), Create(l3), Create(l4))
        self.play(FadeIn(beep_def), FadeIn(no_beep_def), FadeIn(beep_hlth), FadeIn(no_beep_hlth))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(line_colors[3]))
        
        # Highlight 'Beep' nodes
        circle1 = Circle(color="#FF0000", radius=0.45).move_to(self.grid["C1"])
        circle2 = Circle(color="#FF0000", radius=0.45).move_to(self.grid["C4"])
        
        self.play(Create(circle1), Create(circle2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(line_colors[4]))
        
        # Comparison text adjusted per Issue 44
        comparison_text = Text("49.5 > 9.9 !", font_size=24, color="#FF0000")
        self.place_in_area(comparison_text, "D1", "D6", scale_factor=0.9)
        
        # Calculation: P(Def|Beep)
        calc_formula = MathTex(
            r"P(\text{Defective} | \text{Beep}) = \frac{9.9}{9.9 + 49.5} \approx 16.7\%",
            color="#FF0000", font_size=28
        )
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").scale(0.5)
        
        calc_group = VGroup(calc_formula, robot_icon).arrange(RIGHT, buff=0.5)
        self.place_in_area(calc_group, "E1", "F6", scale_factor=0.9)
        
        self.play(FadeIn(comparison_text))
        self.play(Write(calc_formula), FadeIn(robot_icon))
        self.wait(3)
