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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Rules of the Game: Axiomatic Definition", 
                          ["Vector spaces require eight axioms.", 
                           "Closure under addition is vital.", 
                           "Closure under scalar multiplication applies."])
        
        # UI Elements (using checklist icon)
        checklist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/checklist.svg")
        checklist_icon.set_color(WHITE)
        
        axioms = VGroup(
            Text("1. Closure (+)", font_size=20),
            Text("2. Closure (sc)", font_size=20),
            Text("3. Commutativity", font_size=20),
            Text("4. Associativity", font_size=20),
            Text("5. Zero Vector", font_size=20),
            Text("6. Add. Inverses", font_size=20),
            Text("7. Unit Scalar", font_size=20),
            Text("8. Distributivity", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        
        checklist_content = VGroup(checklist_icon, axioms).arrange(RIGHT, buff=0.2)
        
        # Place axioms in B4-E6 as requested
        self.place_in_area(checklist_content, 'B4', 'E6', scale_factor=0.7)
        
        # Add checkmarks to be revealed
        checkmarks = VGroup(*[\
            Checkmark().set_color("#00FF00").scale(0.3).next_to(axioms[i], RIGHT, buff=0.2)
            for i in range(len(axioms))
        ])
        for cm in checkmarks:
            cm.set_opacity(0)
        self.add(checkmarks)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        self.play(FadeIn(checklist_content))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color("#FF4500"))
        self.play(checkmarks[0].animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color("#FF4500"))
        self.play(checkmarks[1].animate.set_opacity(1))
        self.wait(2)

class Checkmark(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly([
            UP * 0.1 + LEFT * 0.2,
            ORIGIN,
            DOWN * 0.2 + RIGHT * 0.3
        ])
        self.set_stroke(width=6)
