from manim import *
import numpy as np

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
        self.setup_layout("Real-World Application: Data Locality", 
                          ["Space-filling curves preserve crucial spatial locality.", 
                           "Map multi-dimensional data into 1D databases.", 
                           "Hilbert search outperforms inefficient row-major scans."])
        
        # --- Assets ---
        # Note: SVG files are assumed to be available at the paths provided.
        # Replacing abstract logic with required asset loading.
        memory_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/memory.svg", color=WHITE)
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#AAAAAA")
        # Visual Critic 30: Adjust positioning to avoid title.
        self.place_in_area(memory_icon, 'B3', 'B6', scale_factor=0.75)
        self.play(FadeIn(memory_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFAA55")
        grid_map = VGroup(*[Square(side_length=0.6, color=WHITE) for _ in range(9)])
        grid_map.arrange_in_grid(3, 3, buff=0.1)
        
        # Visual Critic 32: Positioning grid map lower to avoid connector overlap.
        self.place_in_area(grid_map, 'E1', 'F6', scale_factor=0.7)
        self.play(Create(grid_map))
        
        # Mapping lines
        arrows = VGroup()
        for i in range(min(3, len(grid_map))):
            arrows.add(Line(memory_icon.get_bottom(), grid_map[i].get_top(), color="#FFAA55"))
        self.play(Create(arrows))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#55AAFF")
        # Add the computer icon
        self.place_in_area(computer_icon, 'D4', 'D6', scale_factor=0.5)
        self.play(FadeIn(computer_icon))
        
        # Highlight cache locality
        highlight = SurroundingRectangle(grid_map[0:3], color="#55AAFF", buff=0.05)
        self.play(Create(highlight))
        self.wait(2)
