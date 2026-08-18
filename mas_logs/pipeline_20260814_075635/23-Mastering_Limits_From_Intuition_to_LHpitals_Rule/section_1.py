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
        self.setup_layout("The Intuition of Limits: The Squirrel's Path", [
            "The squirrel approaches the tree hollow at x=1.",
            "The path y=(x²-1)/(x-1) shows a missing point.",
            "Height 2 is the destination, the limit L."
        ])
        
        # Axes for the graph
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.5)
        self.add(axes)
        
        # Curve y = (x^2-1)/(x-1) = x+1 for x != 1
        curve = axes.plot(lambda x: x + 1, x_range=[0, 2.5], color=WHITE)
        self.add(curve)
        
        # Squirrel asset
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg", color="#FFD700")
        self.place_at_grid(squirrel, 'E2', scale_factor=0.7)
        
        # Tree asset
        tree = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tree.svg", color="#32CD32")
        self.place_at_grid(tree, 'B3', scale_factor=0.5)
        
        # Hollow at (1, 2)
        hollow = Circle(radius=0.08, color="#FF4500", fill_opacity=0).move_to(axes.c2p(1, 2))
        self.add(hollow)
        
        missing_point_label = Text("Hole", font_size=20, color="#FF4500")
        self.place_at_grid(missing_point_label, 'D3', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(FadeIn(squirrel))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        self.play(FadeIn(missing_point_label), Create(hollow))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(FadeIn(tree))
        self.wait(1)
