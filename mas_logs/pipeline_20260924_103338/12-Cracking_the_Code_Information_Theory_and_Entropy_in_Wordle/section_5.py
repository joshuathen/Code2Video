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
        self.setup_layout("Wrap-up: From Wordle to Reality", [
            "Entropy helps us compress data efficiently.", 
            "Medical diagnosis uses these principles to narrow tests.", 
            "This information theory applies everywhere in life."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Entropy helps us compress data efficiently.
        self.lecture[0].set_color("#FFD700")
        
        world_map = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        self.place_in_area(world_map, 'B4', 'E6', scale_factor=0.6)
        self.play(FadeIn(world_map))
        self.wait(4)

        # === Animation for Lecture Line 2 ===
        # Medical diagnosis uses these principles to narrow tests.
        self.lecture[1].set_color("#00FF00")
        
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_icon, 'B5', scale_factor=0.5)
        
        label1 = Text("Data Compression", font_size=18, color=WHITE)
        label2 = Text("Decision Trees", font_size=18, color=WHITE)
        self.place_at_grid(label1, 'B4', scale_factor=0.6)
        self.place_at_grid(label2, 'D4', scale_factor=0.6)
        self.play(FadeIn(computer_icon), FadeIn(label1), FadeIn(label2))
        self.wait(4)

        # === Animation for Lecture Line 3 ===
        # This information theory applies everywhere in life.
        self.lecture[2].set_color("#FF6347")
        
        clipboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clipboard.svg")
        summary = Text("Summary Conclusion", font_size=20, color=WHITE)
        self.place_at_grid(clipboard_icon, 'E5', scale_factor=0.5)
        self.place_at_grid(summary, 'E4', scale_factor=0.6)
        self.play(FadeIn(clipboard_icon), FadeIn(summary))
        self.wait(4)
