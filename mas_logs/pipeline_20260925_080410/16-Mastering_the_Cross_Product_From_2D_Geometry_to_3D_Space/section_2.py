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
        self.setup_layout("The 2D Pseudo-Cross Product", [
            "2D cross product uses a scalar.", 
            "Calculate the signed area of parallelograms.", 
            "It indicates orientation, clockwise or counter-clockwise."
        ])
        
        # Elements
        axes = Axes(x_length=4, y_length=4, x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        v1 = Vector([1, 1], color=WHITE)
        v2 = Vector([-0.5, 1.5], color=WHITE)
        label1 = MathTex(r"\\vec{v}_1", color=WHITE).next_to(v1.get_end(), UP)
        label2 = MathTex(r"\\vec{v}_2", color=WHITE).next_to(v2.get_end(), UP)
        
        # Parallelogram defined by vectors
        para = Polygon(ORIGIN, v1.get_end(), v1.get_end() + v2.get_end(), v2.get_end(), color="#FFA500", fill_opacity=0.3)
        orientation_text = Text("Orientation", color="#00FFFF", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(axes, 'D4', scale_factor=0.8)
        self.play(Write(axes))
        # v1 and v2 need to be associated with the axes space properly. 
        # Using a VGroup to combine vectors and labels for easier placement.
        vectors = VGroup(v1, v2, label1, label2)
        vectors.move_to(axes.get_center())
        
        self.play(Create(v1), Create(v2), Write(label1), Write(label2))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(para, 'D3', 'E5', scale_factor=0.6)
        self.play(Create(para))
        self.lecture[1].set_color("#FFA500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(orientation_text, 'E4', scale_factor=0.9)
        self.play(Write(orientation_text))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
