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
        self.setup_layout("The Zero Case: Collapsed Dimensions", [
            "A zero determinant means space has collapsed.",
            "The transformation squashes shapes into lines.",
            "Original dimensions are lost forever."
        ])
        
        # Placeholder asset loading (The assets provided were none.svg, which doesn't exist. 
        # Using Dot as a placeholder for the required icon presence to satisfy the constraint.)
        icon = Dot(color=WHITE) 
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        title_text = Text("Zero Determinant", font_size=36, color="#FFFFFF")
        self.place_at_grid(title_text, 'A3', scale_factor=0.8)
        self.place_at_grid(icon, 'A5', scale_factor=0.5)
        self.play(FadeIn(title_text), FadeIn(icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        vec1 = Vector([1, 1], color="#3498DB")
        vec2 = Vector([2, 2], color="#3498DB")
        self.place_at_grid(vec1, 'C2')
        self.place_at_grid(vec2, 'C5', scale_factor=0.9)
        area_label = Text("Area=0", font_size=24, color="#3498DB")
        self.place_at_grid(area_label, 'D3', scale_factor=0.8)
        self.play(Create(vec1), Create(vec2), Write(area_label))
        
        # Animate collapse
        self.lecture[2].set_color("#E74C3C")
        target_vec2 = Vector([0.5, 0.5], color="#E74C3C")
        self.place_at_grid(target_vec2, 'D5', scale_factor=0.9)
        
        self.play(
            Transform(vec2, target_vec2),
            area_label.animate.set_color("#F1C40F"),
            run_time=2
        )
        self.play(Flash(area_label, color="#F1C40F"))

        self.wait(2)
