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
        self.setup_layout("Prerequisites: The Vector Space Concept", 
                          ["Computers represent words as lists of numbers.", 
                           "These are vectors in a multi-dimensional space.", 
                           "Similar words cluster together in this space."])
        
        king_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg")
        queen_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/queen.svg")

        # === Animation for Lecture Line 1 ===
        vector_text = Text("King: [0.9, 0.1, 0.8]", font_size=24, color="#FF5733")
        self.place_at_grid(vector_text, 'B2', scale_factor=0.8)
        self.place_at_grid(king_icon, 'B4', scale_factor=0.5)
        self.play(FadeIn(vector_text), FadeIn(king_icon))
        self.play(self.lecture[0].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 2 ===
        axes = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-2, 2, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, 'D1', 'F6', scale_factor=0.4)
        vector_arrow = Arrow(ORIGIN, [0.8, 0.7, 0.5], color=YELLOW)
        axes.add(vector_arrow)
        self.play(Create(axes), GrowArrow(vector_arrow))
        self.play(self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        king_circle = Circle(radius=0.5, color="#33FF57")
        queen_circle = Circle(radius=0.5, color="#33FF57")
        self.place_at_grid(king_circle, 'D4')
        self.place_at_grid(queen_circle, 'E5')
        self.place_at_grid(king_icon.copy(), 'D4', scale_factor=0.3)
        self.place_at_grid(queen_icon, 'E5', scale_factor=0.3)
        
        self.play(Create(king_circle), Create(queen_circle))
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        self.wait(2)
