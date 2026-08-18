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
        self.setup_layout("Application: Feature Detection", ["Kernels act as pattern detectors.", "Positive and negative values identify edges.", "High contrast yields strong signal response."])
        
        # Reveal lecture lines sequentially
        self.play(FadeIn(self.lecture[0]))
        
        # === Animation for Lecture Line 1: Kernels act as pattern detectors ===
        self.lecture[0].set_color("#888888")
        # Load asset
        img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        img.set_color("#888888")
        self.place_in_area(img, 'B4', 'C6', scale_factor=0.5)
        self.play(FadeIn(img))

        # === Animation for Lecture Line 2: Positive and negative values identify edges ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF5500")
        
        kernel = Matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], v_buff=0.3, h_buff=0.3).set_color("#FF5500")
        self.place_at_grid(kernel, 'D5', scale_factor=0.6)
        self.play(FadeIn(kernel))

        # === Animation for Lecture Line 3: High contrast yields strong signal response ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Load asset
        edge_detection = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        edge_detection.set_color("#00FF00")
        self.place_at_grid(edge_detection, 'D2', scale_factor=0.7)
        self.play(FadeIn(edge_detection))
        
        self.wait(2)
