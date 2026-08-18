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
        self.setup_layout("The Proof Logic: Connecting Vertices", [
            "Define a function based on point-pair distance.",
            "Construct squares using midpoint and distance.",
            "The non-square configurations form an inconsistent set.",
            "This logic forces a perfect square solution.",
            "The square snaps onto the curve's perimeter."
        ])
        
        # Curve
        curve = ParametricFunction(
            lambda t: np.array([0.8 * np.cos(t) + 0.3 * np.sin(3*t), 0.8 * np.sin(t), 0]),
            t_range=[0, 2*PI], color=WHITE
        )
        self.place_in_area(curve, 'B3', 'E6', scale_factor=1.2)
        self.add(curve)
        
        # Square Asset
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        self.place_at_grid(square, 'C4', scale_factor=0.9)
        square.set_color(YELLOW)
        self.add(square)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 3 ===
        # Create an animation group to be placed
        anim_group = VGroup(Dot(color=RED), Dot(color=BLUE)).arrange(RIGHT)
        self.place_in_area(anim_group, 'B4', 'E6', scale_factor=1.0)
        self.play(Create(anim_group))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 4 ===
        self.play(square.animate.set_color("#FF00FF"))
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 5 ===
        self.play(square.animate.set_color("#00FFFF").move_to(self.grid['D4']))
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
