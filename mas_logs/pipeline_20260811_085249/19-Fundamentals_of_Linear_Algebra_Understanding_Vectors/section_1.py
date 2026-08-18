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
        lecture_lines = [
            "Vectors are arrows representing direction and magnitude.",
            "They can also be lists of numbers.",
            "Think of them as positions in space."
        ]
        self.setup_layout("Introduction: What is a Vector?", lecture_lines)
        
        # Setup Axes
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": True}).scale(0.5)
        # Apply fix for VideoCritic issue 21
        self.place_in_area(axes, 'D2', 'F6', scale_factor=0.6)
        self.add(axes)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        self.place_at_grid(compass, 'B2', scale_factor=0.3)
        self.place_at_grid(ruler, 'B4', scale_factor=0.3)

        # Animation Elements
        vector = Vector([2, 1], color="#FF0000")
        label = Text("Vector v", font_size=20, color="#FF0000")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.play(Create(vector), FadeIn(compass))
        label.next_to(vector.get_end(), UP, buff=0.1)
        self.play(Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        
        new_vec = Vector([3, 2], color="#00FF00")
        # VideoCritic issue 22 suggests using variable name 'vector_label'
        vector_label = Text("v = (x, y)", font_size=20, color="#00FF00")
        self.place_at_grid(vector_label, 'D1', scale_factor=0.7)
        
        # Construct group to place as per issue 23
        vector_group = VGroup(new_vec, vector_label)
        self.place_in_area(vector_group, 'C3', 'E5', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(vector, new_vec),
            ReplacementTransform(label, vector_label),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(
            FadeIn(ruler),
            new_vec.animate.shift(RIGHT * 0.5 + UP * 0.5),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.play(Indicate(new_vec))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
