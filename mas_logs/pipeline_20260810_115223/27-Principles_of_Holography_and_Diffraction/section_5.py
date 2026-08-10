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
        self.setup_layout("Summary and Real-world Application", [
            "Diffraction records data; interference reconstructs reality.", 
            "Used in security features like credit cards.", 
            "Enables high-density data storage."
        ])
        
        # === Animation for Lecture Line 1 ===
        rect1 = Rectangle(height=1.5, width=2.5, color=WHITE).set_fill(BLUE, opacity=0.3)
        rect2 = Rectangle(height=1.5, width=2.5, color=WHITE).set_fill(GREEN, opacity=0.3)
        
        labels = VGroup(
            Text("Recording", font_size=20),
            Text("Reconstruction", font_size=20)
        )
        
        group1 = VGroup(rect1, labels[0]).arrange(DOWN)
        group2 = VGroup(rect2, labels[1]).arrange(DOWN)
        
        self.place_in_area(group1, "B2", "C3")
        self.place_in_area(group2, "D4", "E5", scale_factor=0.7)
        
        self.play(Flash(group1), Flash(group2))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/creditcard.svg
        card = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/creditcard.svg")
        card_label = Text("Credit Card Hologram", font_size=20, color="#00FF00")
        
        card_group = VGroup(card, card_label).arrange(DOWN)
        self.place_in_area(card_group, "B4", "D5", scale_factor=0.6)
        
        self.play(Create(card_group), run_time=2)
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        storage_list = VGroup(
            Text("Security", font_size=24),
            Text("Data Storage", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_in_area(storage_list, "E1", "F6", scale_factor=0.5)
        self.play(Write(storage_list))
        self.lecture[2].set_color(WHITE)
