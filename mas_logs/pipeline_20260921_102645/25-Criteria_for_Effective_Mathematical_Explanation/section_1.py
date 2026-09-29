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
            "Effective explanation builds a bridge to understanding.",
            "Confusion is the starting point for learning.",
            "Mastery is the target across the chasm.",
            "Build cognitive clarity between these two points.",
            "Let's start the journey now."
        ]
        self.setup_layout("Introduction: The 'Bridge' Analogy", lecture_lines)
        
        # Elements
        confusion = Square(side_length=1.0, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(confusion, 'B1')
        label_c = Text("Confusion", font_size=16).next_to(confusion, DOWN)
        
        mastery = Square(side_length=1.0, color=WHITE, fill_opacity=0.5)
        self.place_at_grid(mastery, 'B6')
        label_m = Text("Mastery", font_size=16).next_to(mastery, DOWN)
        
        # Asset: bridge.svg
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_in_area(bridge_icon, 'B2', 'B5', scale_factor=0.8)
        
        bridge_line = DashedLine(confusion.get_right(), mastery.get_left(), color=YELLOW)
        bridge_text = Text("Logical Bridge", font_size=18, color=TEAL)
        self.place_at_grid(bridge_text, 'A3', scale_factor=0.9)
        
        mover = Circle(radius=0.2, color=RED, fill_opacity=1)
        mover.move_to(confusion.get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), Create(confusion), Create(mastery), Write(label_c), Write(label_m))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW), Create(bridge_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW), Create(bridge_line), Write(bridge_text))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW), mover.animate.move_to(mastery.get_center()))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW), 
                  Flash(bridge_icon, color=PURPLE))
        self.wait(1)
