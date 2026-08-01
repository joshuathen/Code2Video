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

class Section5Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout
        title_text = "Real-World Application: Data Mapping"
        lecture_lines = [
            "Hilbert curves map 2D data into 1D memory sequences.",
            "This clustering keeps related spatial data physically close.",
            "This optimization speeds up complex database and map queries."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_MAP_BG = "#444444"
        COLOR_MEM_BG = "#AAAAAA"
        COLOR_L1 = "#FFFF00"  # Yellow
        COLOR_L2 = "#00FFFF"  # Cyan
        COLOR_L3 = "#FF8800"  # Orange

        # === Animation for Lecture Line 1 ===
        # Show a 2D grid 'Map' in #444444 and a 1D horizontal bar 'Memory' in #AAAAAA.
        self.play(self.lecture[0].animate.set_color(COLOR_L1))

        # Create Map (4x4)
        map_cells = VGroup()
        cell_positions = []
        # Logical grid mapping: B1-B4, C1-C4, D1-D4, E1-E4
        grid_rows = ["B", "C", "D", "E"]
        grid_cols = ["1", "2", "3", "4"]
        for r in grid_rows:
            for c in grid_cols:
                pos = self.grid[f"{r}{c}"]
                cell_positions.append(pos)
                sq = Square(side_length=0.9, stroke_color=COLOR_MAP_BG, fill_color=COLOR_MAP_BG, fill_opacity=0.2)
                sq.move_to(pos)
                map_cells.add(sq)
        
        map_label = Text("2D Map Data", font_size=20, color=WHITE)
        self.place_at_grid(map_label, "A2", scale_factor=1.0)

        # Create Memory Bar (1x16 squares)
        memory_blocks = VGroup(*[Square(side_length=0.2, stroke_color=COLOR_MEM_BG, fill_color=COLOR_MEM_BG, fill_opacity=0.2) for _ in range(16)])
        memory_blocks.arrange(RIGHT, buff=0.1)
        self.place_in_area(memory_blocks, "F1", "F6")
        
        memory_label = Text("1D Memory Sequence", font_size=20, color=WHITE)
        self.place_at_grid(memory_label, "E6", scale_factor=1.0)

        self.play(
            FadeIn(map_cells), 
            FadeIn(map_label), 
            FadeIn(memory_blocks), 
            FadeIn(memory_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Trace the Hilbert curve on the Map and highlight corresponding contiguous segments on the Memory bar.
        self.play(self.lecture[1].animate.set_color(COLOR_L2))

        # Hilbert Path Sequence (indices in cell_positions: 0=B1, 1=B2, 2=B3, 3=B4, 4=C1, ...)
        # Path: B1 -> C1 -> C2 -> B2 -> B3 -> B4 -> C4 -> C3 -> D3 -> D4 -> E4 -> E3 -> E2 -> D2 -> D1 -> E1
        path_indices = [0, 4, 5, 1, 2, 3, 7, 6, 10, 11, 15, 14, 13, 9, 8, 12]
        
        curve_segments = VGroup()
        for i in range(len(path_indices)):
            # Memory highlight animation
            mem_anim = memory_blocks[i].animate.set_fill(COLOR_L2, opacity=0.8).set_stroke(COLOR_L2)
            
            if i > 0:
                seg = Line(cell_positions[path_indices[i-1]], cell_positions[path_indices[i]], color=COLOR_L2, stroke_width=4)
                curve_segments.add(seg)
                self.play(Create(seg), mem_anim, run_time=0.2)
            else:
                self.play(mem_anim, run_time=0.2)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight a square region on the Map; show its data points clustering together in Memory.
        self.play(self.lecture[2].animate.set_color(COLOR_L3))

        # Regional Cluster (e.g., top-right quadrant): B3, B4, C3, C4 (Indices 2, 3, 6, 7 in cell_positions)
        region_sq_indices = [2, 3, 6, 7]
        region_highlights = VGroup(*[map_cells[idx].copy().set_fill(COLOR_L3, opacity=0.8).set_stroke(COLOR_L3, width=4) for idx in region_sq_indices])
        
        # Memory blocks 4, 5, 6, 7 in the Hilbert sequence correspond to B3, B4, C4, C3
        mem_cluster_indices = [4, 5, 6, 7]
        mem_highlights = VGroup(*[memory_blocks[idx].copy().set_fill(COLOR_L3, opacity=0.8).set_stroke(COLOR_L3, width=4) for idx in mem_cluster_indices])

        self.play(FadeIn(region_highlights), FadeIn(mem_highlights))
        self.play(
            Indicate(region_highlights, color=COLOR_L3), 
            Indicate(mem_highlights, color=COLOR_L3)
        )
        self.wait(2)
