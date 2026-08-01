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
        title = "The Posterior: Calculating the New Ratio"
        lines = [
            "Compare the cat-plus-beep area to total beep area.",
            "The low prior keeps the final probability small.",
            "The new probability is roughly thirty-three percent.",
            "False alarms still occupy a significant portion of space.",
            "This is the visual essence of Bayes' Theorem."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CAT_BEEP = GREEN
        COLOR_TOTAL_BEEP = BLUE_D
        COLOR_FALSE_ALARM = RED_D
        COLOR_EQUATION = WHITE
        COLOR_GOLD = "#F1C40F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Equation Display
        formula = Text(
            "P(A|B) = Area(A ∩ B) / Area(B)",
            font_size=20,
            color=COLOR_EQUATION
        )
        self.place_in_area(formula, "A1", "A6")
        
        # Area Visuals
        rect_green = Rectangle(width=1.5, height=2.0, fill_color=COLOR_CAT_BEEP, fill_opacity=0.3, stroke_color=COLOR_CAT_BEEP)
        rect_red = Rectangle(width=3.0, height=2.0, fill_color=COLOR_FALSE_ALARM, fill_opacity=0.3, stroke_color=COLOR_FALSE_ALARM)
        beep_group = VGroup(rect_green, rect_red).arrange(RIGHT, buff=0)
        rect_outline = Rectangle(width=4.5, height=2.0, color=COLOR_TOTAL_BEEP, stroke_width=2)
        
        # Center the outline on the grouped rectangles
        rect_outline.move_to(beep_group.get_center())
        
        viz_group = VGroup(beep_group, rect_outline)
        self.place_in_area(viz_group, "C1", "E6")
        
        self.play(Write(formula))
        self.play(FadeIn(viz_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Prior note (Issue 38)
        prior_note = Text("Low Prior: P(Cat) = 1/10", font_size=20, color=COLOR_GOLD)
        self.place_in_area(prior_note, 'B2', 'B5', scale_factor=0.8)
        
        # Highlight green area and show 0.09
        label_09 = Text("0.09", font_size=22, color=COLOR_CAT_BEEP)
        self.place_at_grid(label_09, "D2") # Near green part
        
        formula_num = Text("P(A|B) = 0.09 / Area(B)", font_size=20, color=COLOR_EQUATION)
        self.place_in_area(formula_num, "A1", "A6")
        
        self.play(rect_green.animate.set_fill(opacity=0.8))
        self.play(Write(label_09), Write(prior_note), Transform(formula, formula_num))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight total area and show 0.27
        label_27 = Text("0.27", font_size=22, color=COLOR_TOTAL_BEEP)
        self.place_at_grid(label_27, "D6") # Near right side of blue
        
        formula_den = Text("P(A|B) = 0.09 / 0.27", font_size=20, color=COLOR_EQUATION)
        self.place_in_area(formula_den, "A1", "A6")
        
        # Incorporating cat icon (Issue 26)
        cat_icon = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png").scale(0.3)
        p_text_1 = Text("P(", font_size=24, color=COLOR_GOLD)
        p_text_2 = Text("|Beep) ≈ 33%", font_size=24, color=COLOR_GOLD)
        calc_result = Group(p_text_1, cat_icon, p_text_2).arrange(RIGHT, buff=0.1)
        # Position result (Issue 39)
        self.place_in_area(calc_result, 'A2', 'A5', scale_factor=0.7)
        
        self.play(rect_outline.animate.set_stroke(width=5))
        self.play(Write(label_27), Transform(formula, formula_den))
        self.play(FadeOut(formula), FadeIn(calc_result))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # False alarm label (Issue 40)
        false_alarm_label = Text("False Alarms", font_size=20, color=COLOR_FALSE_ALARM)
        self.place_in_area(false_alarm_label, 'F4', 'F6', scale_factor=0.8)
        
        self.play(rect_red.animate.set_fill(opacity=0.6))
        self.play(Write(false_alarm_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Circle the 33% result
        emphasis_circle = Circle(color=COLOR_GOLD, stroke_width=4)
        emphasis_circle.surround(calc_result, buffer_factor=1.2)
        
        self.play(Create(emphasis_circle))
        self.wait(2)
