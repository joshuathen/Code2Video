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
        self.setup_layout("Kernel Personalities", [
            "Different kernels change the image appearance.",
            "Blur kernels compute local averages.",
            "Edge detection kernels highlight rapid differences."
        ])
        
        # Load assets
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/photograph.svg")
        
        # Create kernel grids
        def create_kernel(values, color):
            grid = VGroup()
            for row in values:
                for val in row:
                    cell = Square(side_length=0.6, fill_opacity=0.3, fill_color=color, stroke_color=WHITE)
                    label = Text(str(val), font_size=18, color=WHITE)
                    cell.add(label)
                    grid.add(cell)
            grid.arrange_in_grid(rows=3, cols=3, buff=0)
            return grid

        # Kernel configurations
        k_identity = create_kernel([[0,0,0], [0,1,0], [0,0,0]], WHITE)
        k_blur = create_kernel([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]], "#00BFFF")
        k_edge = create_kernel([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], "#FF4500")

        # Place them
        self.place_at_grid(k_identity, 'B5', scale_factor=0.7)
        self.place_at_grid(k_blur, 'B5', scale_factor=0.7)
        self.place_at_grid(k_edge, 'D5', scale_factor=0.7)
        
        self.place_at_grid(icon1, 'B2', scale_factor=0.5)
        self.place_at_grid(icon2, 'D2', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        k_blur.set_opacity(0)
        k_edge.set_opacity(0)
        icon2.set_opacity(0)
        self.play(FadeIn(k_identity), FadeIn(icon1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(ReplacementTransform(k_identity, k_blur))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(ReplacementTransform(k_blur, k_edge), FadeIn(icon2))
        self.wait(1)
