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
        lecture_lines = ["Holography stores complex diffraction patterns.", "Secures credit card data.", "Enables advanced AR displays."]
        self.setup_layout("Summary and Real-World Application", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Load asset and apply styling
        hologram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        hologram_icon.set_color(RED)
        self.place_at_grid(hologram_icon, 'B3', scale_factor=0.6)
        self.play(FadeIn(hologram_icon), Indicate(hologram_icon), run_time=1.5)
        self.lecture[0].set_color(RED)

        # === Animation for Lecture Line 2 ===
        # Load asset and apply styling
        card_img = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png")
        card_label = Text("Credit Card", font_size=20, color=WHITE).next_to(card_img, DOWN)
        card_group = Group(card_img, card_label)
        card_group.set_color(PURPLE)
        self.place_in_area(card_group, 'D3', 'E5', scale_factor=0.8)
        self.play(FadeIn(card_group), run_time=1.5)
        self.lecture[1].set_color(PURPLE)

        # === Animation for Lecture Line 3 ===
        ar_glass = Ellipse(color=YELLOW, width=2, height=1)
        self.place_at_grid(ar_glass, 'C5', scale_factor=0.6)
        self.play(Create(ar_glass), run_time=1.5)
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
