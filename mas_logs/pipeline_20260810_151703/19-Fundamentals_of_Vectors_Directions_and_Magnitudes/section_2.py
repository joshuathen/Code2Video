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
        self.setup_layout("Visualizing Vector Components", ["We break vectors into components.", "Horizontal movement forms the x-component.", "Vertical movement forms the y-component."])
        
        axes = Axes(x_range=[0, 5], y_range=[0, 5], axis_config={"include_tip": True})
        vector = Vector([3, 4], color=BLUE)
        v_group = VGroup(axes, vector)
        # Apply adjustment: B2-D4
        self.place_in_area(v_group, 'B2', 'D4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), GrowArrow(vector))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        v_x_line = Line(axes.c2p(0, 0), axes.c2p(3, 0), color=RED)
        v_x_label = MathTex(r"v_x", color="#FFFFE0")
        # Apply adjustment: E4
        self.place_at_grid(v_x_label, 'E4', scale_factor=0.8)
        self.play(Create(v_x_line), Write(v_x_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        v_y_line = Line(axes.c2p(3, 0), axes.c2p(3, 4), color=GREEN)
        v_y_label = MathTex(r"v_y", color="#FFFFE0")
        # Apply adjustment: B2
        self.place_at_grid(v_y_label, 'B2', scale_factor=0.8)
        self.play(Create(v_y_line), Write(v_y_label))
        self.wait(2)
