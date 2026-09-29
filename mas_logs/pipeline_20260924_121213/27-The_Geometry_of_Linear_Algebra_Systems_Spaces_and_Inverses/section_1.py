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
        lecture_lines = ["Linear equations represent intersecting geometric spaces.", "Ax=b is finding the intersection point.", "Example: Two lines intersecting in 2D."]
        self.setup_layout("Geometric Interpretation of Linear Systems", lecture_lines)
        
        # Grid axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.8)
        self.add(axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        line1 = axes.plot(lambda x: x + 1, color="#00FFFF")
        line2 = axes.plot(lambda x: -x + 1, color="#FF00FF")
        label1 = Tex("L1", color=WHITE).next_to(line1.point_from_proportion(0.8), UP)
        label2 = Tex("L2", color=WHITE).next_to(line2.point_from_proportion(0.2), UP)
        self.play(Create(line1), Create(line2), Write(label1), Write(label2))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        intersection = Dot(axes.c2p(0, 1), color="#FFFF00", radius=0.1)
        # Using placeholder SVG for asset
        intersection_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.5).next_to(intersection, RIGHT)
        label_int = Text("Intersection", font_size=16, color=WHITE).next_to(intersection_icon, RIGHT)
        self.play(Flash(intersection), Create(intersection), Write(intersection_icon), Write(label_int))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#32CD32")
        vec_b = Arrow(axes.c2p(-2, -1), axes.c2p(0, 1), color="#32CD32")
        self.play(GrowArrow(vec_b))
        self.wait(2)
