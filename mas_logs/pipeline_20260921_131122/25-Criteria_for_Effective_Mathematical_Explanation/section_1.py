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
        lecture_lines = [
            "Mathematical explanation bridges formal proof and intuition.",
            "Accuracy ensures foundation is sound.",
            "Logical flow guides understanding.",
            "Cognitive economy simplifies complexity.",
            "These pillars build effective explanations."
        ]
        self.setup_layout("Introduction: The Goal of Mathematical Clarity", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display text: 'Mathematical Explanation = Proof + Intuition'
        formula = Text("Explanation = Proof + Intuition", font_size=20)
        proof = Rectangle(width=1.5, height=0.8, color=BLUE).set_fill(BLUE, opacity=0.3)
        intuition = Rectangle(width=1.5, height=0.8, color=GREEN).set_fill(GREEN, opacity=0.3)
        nexus = Circle(radius=0.3, color=YELLOW).set_fill(YELLOW, opacity=0.3)
        
        math_components = VGroup(formula, proof, intuition, nexus)
        self.place_in_area(math_components, 'B3', 'C5', scale_factor=0.8)
        
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")
        
        # Animate a line from a 'Proof' box to an 'Intuition' box meeting at a center 'Understanding' nexus.
        self.play(Create(proof), Create(intuition), Create(nexus))
        line1 = Line(proof.get_right(), nexus.get_left())
        line2 = Line(intuition.get_left(), nexus.get_right())
        self.play(Create(line1), Create(line2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display text: 'Three Pillars:' in #FFFF00.
        pillars_text = Text("Three Pillars:", font_size=24, color="#FFFF00")
        self.place_at_grid(pillars_text, 'A3', scale_factor=1.0)
        self.play(FadeIn(pillars_text))
        self.lecture[1].set_color("#FF0000")
        
        # === Animation for Lecture Line 3 ===
        # Fade in three color-coded boxes: 'Accuracy' (#FF0000), 'Logical Flow' (#00FF00), 'Cognitive Economy' (#0000FF).
        box1 = Text("Accuracy", font_size=18, color="#FF0000")
        box2 = Text("Logical Flow", font_size=18, color="#00FF00")
        box3 = Text("Cognitive Economy", font_size=18, color="#0000FF")
        
        group_boxes = VGroup(box1, box2, box3).arrange(RIGHT)
        self.place_in_area(group_boxes, 'D2', 'E6', scale_factor=0.7)
        
        self.play(FadeIn(box1), FadeIn(box2), FadeIn(box3))
        self.lecture[2].set_color("#00FF00")
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#0000FF")
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Animate a [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg] drawing itself across the screen connecting the pillars.
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color=WHITE)
        self.place_at_grid(bridge, 'E4', scale_factor=0.8)
        self.play(Create(bridge))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
