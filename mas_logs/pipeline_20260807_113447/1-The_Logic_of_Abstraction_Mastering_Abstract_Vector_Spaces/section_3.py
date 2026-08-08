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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The Axiom Club has strict rules.",
            "Eight formal axioms determine membership.",
            "Polynomials check every required box.",
            "Rules ensure consistent mathematical behavior.",
            "Structure transforms sets into spaces."
        ]
        self.setup_layout("The 'Axiom Club' Rules", lecture_lines)
        
        # Colors - Using standard Manim CE color constants
        color1 = WHITE
        color2 = PINK
        color3 = BLUE
        color4 = GREEN
        color5 = YELLOW
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        # 8 empty checkboxes on the left side of the visual area
        boxes = VGroup()
        box_locs = [
            'A1', 'B1', 'C1', 'D1', 'E1', 'F1',  # Axioms 1-6
            'A3', 'B3'                           # Axioms 7-8
        ]
        for loc in box_locs:
            box = Square(side_length=0.4, color=WHITE)
            self.place_at_grid(box, loc)
            boxes.add(box)
        
        self.play(Create(boxes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Labels for checkboxes
        labels = VGroup()
        label_texts = [
            "Assoc(+)", "Comm(+)", "Id(+)", "Inv(+)",
            "ScalAssoc", "ScalId", "Dist1", "Dist2"
        ]
        label_locs = [
            'A2', 'B2', 'C2', 'D2', 'E2', 'F2',
            'A4', 'B4'
        ]
        for text, loc in zip(label_texts, label_locs):
            lbl = Text(text, font_size=16, color=color2)
            self.place_at_grid(lbl, loc)
            labels.add(lbl)
            
        self.play(Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Polynomial expression on the right
        poly = MathTex("p(x) = a_n x^n + \\dots + a_0", color=color3)
        # Position in the lower right area
        self.place_in_area(poly, 'C5', 'E6', scale_factor=0.8)
        
        self.play(Write(poly))
        
        # Check Commutativity and Identity (Axioms 2 and 3)
        check2 = MathTex("\\checkmark", color=GREEN).scale(0.8).move_to(boxes[1])
        check3 = MathTex("\\checkmark", color=GREEN).scale(0.8).move_to(boxes[2])
        
        self.play(
            boxes[1].animate.set_color(GREEN),
            boxes[2].animate.set_color(GREEN),
            Create(check2),
            Create(check3)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color4))
        
        # All remaining checkboxes turn green and check
        remaining_checks = VGroup()
        remaining_indices = [0, 3, 4, 5, 6, 7]
        for i in remaining_indices:
            chk = MathTex("\\checkmark", color=GREEN).scale(0.8).move_to(boxes[i])
            remaining_checks.add(chk)
            
        self.play(
            *[boxes[i].animate.set_color(GREEN) for i in remaining_indices],
            Create(remaining_checks),
            *[Flash(box, color=GREEN, flash_radius=0.3) for box in boxes]
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color5))
        
        # 'CLUB MEMBER' appears over the polynomial
        member_text = Text("CLUB MEMBER", font_size=36, color=color5, weight=BOLD)
        # Position above the polynomial area
        self.place_in_area(member_text, 'B5', 'B6', scale_factor=0.7)
        
        self.play(FadeIn(member_text, shift=UP))
        self.play(Indicate(member_text))
        self.wait(2)
