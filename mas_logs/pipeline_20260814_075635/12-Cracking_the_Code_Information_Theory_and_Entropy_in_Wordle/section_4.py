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
        self.setup_layout("The Feedback Loop: Iterative Optimization", [
            "Feedback updates our probability distribution.",
            "Each guess refines the target.",
            "Entropy drops with every new clue."
        ])
        
        # Load asset
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        
        # === Animation for Lecture Line 1 ===
        # Show an arrow cycle representing input-output-feedback. #FFFFFF (text), #00FFFF (arrows)
        loop_group = VGroup(
            Square(side_length=1.5, color=WHITE),
            Arrow(start=UP*1.2, end=DOWN*1.2, color="#00FFFF"),
            Arrow(start=LEFT*1.2, end=RIGHT*1.2, color="#00FFFF"),
            target_icon.copy().set_color(WHITE)
        )
        self.place_in_area(loop_group, 'A3', 'C6', scale_factor=0.6)
        self.play(Create(loop_group), self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Highlight the 'update' phase in the loop with bolding. #FFFF00
        self.play(loop_group[0].animate.set_color("#FFFF00"), self.lecture[1].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        # Visualize the model accuracy line rising over iterations. #00FF00 (line)
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 1, 0.25], x_length=3, y_length=2).scale(0.7)
        graph = axes.plot(lambda x: 1 - (0.5)**x, color="#00FF00")
        target_marker = target_icon.copy().set_color("#00FF00").scale(0.3)
        target_marker.move_to(axes.c2p(4, 1))
        
        accuracy_group = VGroup(axes, graph, target_marker)
        self.place_in_area(accuracy_group, 'D3', 'F6', scale_factor=0.6)
        
        self.play(Create(accuracy_group), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
