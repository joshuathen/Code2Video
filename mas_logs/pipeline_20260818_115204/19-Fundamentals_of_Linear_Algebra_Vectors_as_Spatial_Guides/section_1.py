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
        lecture_lines = ["Vectors are arrows, not just single points.", "They represent both magnitude and direction.", "Picture a robot starting at origin (0,0).", "It moves to position (3,2).", "The vector [3,2] marks this displacement."]
        self.setup_layout("Defining the Vector: Movement in Space", lecture_lines)
        
        # Create axes for visualization
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_numbers": True}).scale(0.5)
        # Apply layout fixes from issues
        self.place_in_area(axes, 'C3', 'F5', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        dot = Dot(color=YELLOW).move_to(axes.c2p(0, 0))
        self.add(dot)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        arrow = Arrow(start=axes.c2p(0, 0), end=axes.c2p(3, 2), color=YELLOW, buff=0)
        self.play(Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        robot = Circle(radius=0.15, color=WHITE, fill_opacity=1).move_to(axes.c2p(0, 0))
        self.add(robot)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        self.play(robot.animate.move_to(axes.c2p(3, 2)))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        vector_label = MathTex(r"\\vec{v} = [3, 2]").set_color(PURPLE)
        # Apply layout fix from issues
        self.place_at_grid(vector_label, 'C2', scale_factor=0.7)
        self.play(Write(vector_label))
        self.wait(2)
