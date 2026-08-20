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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", [
            "Vectors are points, duals are tools.",
            "Dot product bridges geometry and functions.",
            "It translates operations into states."
        ])
        
        # Visual objects
        point = Dot(color=BLUE)
        point_label = Text("Point", font_size=18, color=BLUE)
        
        tool = VGroup(Line(ORIGIN, UP*0.3), Line(ORIGIN, RIGHT*0.3)).set_color(GREEN)
        tool_label = Text("Dual Tool", font_size=18, color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(point, 'C2', scale_factor=0.8)
        self.place_at_grid(point_label, 'C1', scale_factor=0.6)
        self.place_at_grid(tool, 'C5', scale_factor=0.8)
        self.place_at_grid(tool_label, 'C6', scale_factor=0.6)
        self.add(point, point_label, tool, tool_label)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        bridge = Arrow(self.grid['C2'], self.grid['C5'], color=RED)
        self.add(bridge)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        state_label_group = Text("Operational State", font_size=20, color=YELLOW)
        self.place_in_area(state_label_group, 'E3', 'E5', scale_factor=0.7)
        self.add(state_label_group)
        self.wait(2)
