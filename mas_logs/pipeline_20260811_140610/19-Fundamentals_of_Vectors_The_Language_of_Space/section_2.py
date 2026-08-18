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
        self.setup_layout("Visualizing Vectors in 2D Space", ["Vectors are arrows on a plane.", "Each starts at the origin point.", "Coordinates define the vector's position."])
        
        # Grid visual
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 4, 1], x_length=4, y_length=3.2, axis_config={"include_numbers": False}).set_color("#333333")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg' is effectively a placeholder, adding as requested but it won't be visible
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color=BLACK)
        
        vector = Vector([3, 2], color="#3498DB")
        origin_dot = Dot(color="#FFFFFF").scale(0.8)
        
        visual_group = VGroup(axes, vector, origin_dot, icon)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#3498DB")
        self.place_in_area(visual_group, 'B3', 'E5', scale_factor=1.2)
        self.play(Create(axes), Create(vector))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        self.add(origin_dot)
        origin_dot.move_to(axes.c2p(0, 0))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        label = MathTex(r"(3, 2)", color="#3498DB").scale(0.7 * 0.8) # B020
        # B011: Place at grid coordinate then next_to
        label_anchor = Dot(axes.c2p(3, 2), fill_opacity=0)
        self.place_at_grid(label_anchor, 'E4', scale_factor=0.9)
        label.next_to(label_anchor, UR, buff=0.1)
        self.play(Write(label))
        self.wait(2)
