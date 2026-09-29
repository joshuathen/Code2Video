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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Reduction bridges intuition and rigorous formalization.",
            "If n is complex, test n equals one.",
            "Sequence visualizers show patterns from low-level data.",
            "Map complex cases to manageable smaller ones.",
            "Testing strength weights."
        ]
        self.setup_layout("Applying the 'Small Case' Heuristic", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#87CEEB"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        dot1 = Dot(color="#32CD32").scale(2)
        self.place_at_grid(dot1, 'B4', scale_factor=0.8)
        self.play(FadeIn(dot1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        dot2 = Dot(color="#FFD700").scale(2)
        dot3 = Dot(color="#FFD700").scale(2)
        self.place_at_grid(dot2, 'B5', scale_factor=0.8)
        self.place_at_grid(dot3, 'B6', scale_factor=0.8)
        self.play(FadeIn(dot2), FadeIn(dot3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF6347"))
        line = Line(start=self.grid['B4'], end=self.grid['B6'], color="#FF6347")
        self.play(Create(line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        explorer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/explorer.svg", color="#FF4500")
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color="#FF4500")
        
        explorer_group = VGroup(explorer, bridge).arrange(RIGHT)
        
        self.place_in_area(explorer_group, 'D4', 'F6', scale_factor=0.7)
        self.play(FadeIn(explorer_group))
        self.wait(2)
