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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Covectors define parallel level lines.", "Dot product counts crossed lines.", "It measures change across planes."]
        self.setup_layout("Visualizing Duality via Level Sets", lecture_lines)
        
        # Grid visualizer: Create parallel lines (level sets)
        lines = VGroup()
        for i in range(-3, 4):
            line = Line(start=[-2, 0.5*i, 0], end=[2, 0.5*i, 0], color="#ADFF2F", stroke_width=2)
            lines.add(line)
        lines.rotate(PI/4)
        
        # Integrating Asset (normal vector metaphor tool)
        # B018: Explicitly render and label
        normal_vec = Vector(direction=[0.5, 0.5, 0], color="#FFFF00")
        normal_label = Text("Normal", font_size=16, color="#FFFF00")
        normal_group = VGroup(normal_vec, normal_label).arrange(DOWN)
        
        # Applying requested placement for better layout (B002, B013)
        self.place_in_area(lines, 'C4', 'F6', scale_factor=0.55)
        self.place_at_grid(normal_group, 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(lines), self.lecture[0].animate.set_color("#ADFF2F"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        highlight_line = lines[3].copy().set_stroke(color="#FF4500", width=4)
        self.add(highlight_line)
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(GrowArrow(normal_vec), Write(normal_label), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)
