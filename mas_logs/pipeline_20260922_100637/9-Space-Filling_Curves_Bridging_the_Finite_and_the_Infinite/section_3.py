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
        lecture_lines = [
            "Giuseppe Peano defined this process formally.",
            "He mapped 1D intervals onto 2D squares.",
            "The path becomes increasingly dense.",
            "Every point finds its unique address.",
            "No gaps remain in the limit."
        ]
        self.setup_layout("The Peano Curve Construction", lecture_lines)
        
        # --- Grid visual ---
        # Note: Asset references handled: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Since the assets are '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg', we will use geometric representations as placeholders as per instructions.
        square = Square(side_length=2.0, color=WHITE)
        self.place_in_area(square, 'A3', 'F5', scale_factor=0.6)
        
        # Setup path
        points = [
            self.grid['B2'], self.grid['B3'], self.grid['B4'],
            self.grid['C4'], self.grid['C3'], self.grid['C2'],
            self.grid['D2'], self.grid['D3'], self.grid['D4']
        ]
        path = VMobject(color=WHITE)
        path.set_points_as_corners(points)
        self.place_in_area(path, 'B3', 'F5', scale_factor=0.65)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        path.set_color("#FF6347")
        self.play(Create(path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#1E90FF"))
        path_dots = VGroup(*[Dot(self.grid[p], color="#1E90FF") for p in ['B2', 'B3', 'B4', 'C4', 'C3', 'C2', 'D2', 'D3', 'D4']])
        self.play(FadeIn(path_dots))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#7FFFD4"))
        # Fix label positioning
        labels = VGroup(*[Text(str(i+1), font_size=16, color="#7FFFD4") 
                          for i, p in enumerate(['B2', 'B3', 'B4', 'C4', 'C3', 'C2', 'D2', 'D3', 'D4'])])
        for i, p in enumerate(['B2', 'B3', 'B4', 'C4', 'C3', 'C2', 'D2', 'D3', 'D4']):
             labels[i].move_to(self.grid[p] + UP*0.2)
        
        # Apply scaling constraint
        labels.scale(0.7)
        self.play(Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        highlight = SurroundingRectangle(path, color="#FFD700", buff=0.1)
        self.play(Create(highlight))
        self.wait(2)
