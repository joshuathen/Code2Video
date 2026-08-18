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
        self.setup_layout("Summary & Quick Check", ["Cheat sheet: Key formula summary.", "Product rule uses addition.", "Chain rule uses multiplication."])
        
        # Animations
        # Create cheat sheet checklist
        checklist_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/checklist.svg")
        checklist_content = VGroup(
            Text("Summary Checklist:", font_size=24, color=YELLOW),
            Text("1. Product Rule: (uv)' = u'v + uv'", font_size=20, color=WHITE),
            Text("2. Chain Rule: (f(g(x)))' = f'(g(x))g'(x)", font_size=20, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        checklist = VGroup(checklist_icon, checklist_content).arrange(RIGHT, buff=0.3)
        
        # Applying layout constraints from issues 33, 35, 40
        self.place_in_area(checklist, 'A2', 'C4', scale_factor=0.85)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(checklist))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(checklist_content[1], color=YELLOW))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(checklist_content[2], color=YELLOW))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Success label as per issues 34, 40
        success = Text("Success!", font_size=40, color=WHITE)
        self.place_at_grid(success, 'D3', scale_factor=0.9)
        self.play(Write(success))
        self.wait(2)
