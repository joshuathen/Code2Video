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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout("The Mystery of the Hidden Function", [
            "Some functions isolate y, like y equals x squared.",
            "Others hide y inside, like x squared plus y squared.",
            "Implicit equations keep y tangled with x."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in green
        self.lecture[0].set_color("#55FF55")
        
        # Define and place explicit label and formula
        explicit_label = Text("Explicit", color="#55FF55", font_size=24)
        self.place_at_grid(explicit_label, "A2")
        
        explicit_eq = Text("y = x^2 + 1", color="#55FF55")
        self.place_at_grid(explicit_eq, "B2")
        
        self.play(FadeIn(explicit_label), FadeIn(explicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2 in red
        self.lecture[1].set_color("#FF5555")
        
        # Define and place implicit label and formula
        # Label at C6 and Formula at C5 to keep distance within 1 grid unit
        implicit_label = Text("Implicit", color="#FF5555", font_size=24)
        self.place_at_grid(implicit_label, "C6")
        
        # Use VGroup of Text to isolate 'y' for the next step without requiring LaTeX
        implicit_eq = VGroup(Text("x^2 + "), Text("y"), Text("^2 = 25")).arrange(RIGHT, buff=0.1).set_color("#FF5555")
        self.place_at_grid(implicit_eq, "C5")
        
        self.play(FadeIn(implicit_label), FadeIn(implicit_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in gold to match the box metaphor
        self.lecture[2].set_color("#FFD700")
        
        # 3. Circle the 'y' in the implicit equation with yellow pulsing light
        y_part = implicit_eq[1]
        yellow_circle = Circle(radius=0.35, color="#FFFF00").move_to(y_part.get_center())
        
        self.play(Create(yellow_circle))
        self.play(
            Indicate(y_part, color="#FFFF00"), 
            yellow_circle.animate.scale(1.2), 
            run_time=0.8, 
            rate_func=there_and_back
        )
        
        # 4. Replace 'y' with a gold gift box icon representing hidden x
        gift_box_rect = Square(side_length=0.4, color="#FFD700", fill_opacity=0.8)
        ribbon_h = Line(gift_box_rect.get_left(), gift_box_rect.get_right(), color=BLACK, stroke_width=2)
        ribbon_v = Line(gift_box_rect.get_top(), gift_box_rect.get_bottom(), color=BLACK, stroke_width=2)
        gift_box = VGroup(gift_box_rect, ribbon_h, ribbon_v).move_to(y_part.get_center())
        
        self.play(
            FadeOut(y_part),
            FadeOut(yellow_circle),
            FadeIn(gift_box)
        )
        self.wait(0.5)
        
        # 5. Show 'Chain Rule' text in white above the box
        # Grid B5 is directly above C5
        chain_rule_text = Text("Chain Rule", color="#FFFFFF", font_size=24)
        self.place_at_grid(chain_rule_text, "B5")
        
        self.play(Write(chain_rule_text))
        self.wait(2)
